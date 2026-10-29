"""Dispatch StepSolver operations to SymPy."""

from collections.abc import Callable, Sequence
from typing import cast

import sympy as sp

from stepsolver.ast import (
    Operation,
    Query,
    SolutionDomain,
    Symbol,
)
from stepsolver.errors import BackendError, QueryError
from stepsolver.sympy_conversion import SympyConverter
from stepsolver.sympy_integrals import (
    match_reciprocal_linear_integral,
    match_sine_square_root_integral,
)
from stepsolver.sympy_series import match_alternating_p_series, match_harmonic_sine_series
from stepsolver.sympy_support import (
    expect_arity,
    expect_integer,
    expect_symbol,
    is_object_mapping,
    is_object_sequence,
    is_real_expression,
)

_SQRTDENEST = cast(
    "Callable[[sp.Basic], sp.Basic]",
    getattr(sp, "sqrtdenest"),  # ruff: ignore[get-attr-with-constant] - not in SymPy's public stubs
)


def _denest_numeric_radical(expression: sp.Basic) -> sp.Basic | None:
    """Return a denested value for a numeric square root, when it is safe."""
    if expression.free_symbols or expression.func is not sp.Pow or len(expression.args) != 2:
        return None
    radicand, exponent = expression.args
    if exponent != sp.Rational(1, 2):
        return None
    try:
        denested = _SQRTDENEST(expression)
    except (NotImplementedError, TypeError, ValueError):
        return None
    if denested == expression or denested.free_symbols:
        return None
    if sp.simplify(sp.Pow(denested, 2) - radicand) != sp.Integer(0):
        return None
    if denested.is_positive is not True:
        return None
    return denested


def _simplify_expression(expression: sp.Basic) -> sp.Basic:
    """Simplify rational expressions without expanding ordinary polynomials."""
    if expression.is_rational_function() and sp.denom(sp.together(expression)) != sp.Integer(1):
        return sp.cancel(expression)
    denested = _denest_numeric_radical(expression)
    if denested is not None:
        return denested
    return sp.simplify(expression)


def _fast_basic_antiderivative(
    expression: sp.Basic,
    variable: sp.Symbol,
) -> sp.Basic | None:
    """Recognize a direct antiderivative without a symbolic integration search."""
    if expression.func is sp.sin and expression.args == (variable,):
        return -sp.cos(variable)
    if expression.func is sp.cos and expression.args == (variable,):
        return sp.sin(variable)
    if expression.func is sp.exp and expression.args == (variable,):
        return sp.exp(variable)
    if expression.func is sp.sinh and expression.args == (variable,):
        return sp.cosh(variable)
    if expression.func is sp.cosh and expression.args == (variable,):
        return sp.sinh(variable)
    if expression.func is sp.tan and expression.args == (variable,):
        return -sp.log(sp.cos(variable))
    if expression.func is sp.Pow and expression.args[0] == variable:
        exponent = expression.args[1]
        if exponent != sp.Integer(-1) and not exponent.has(variable):
            return variable ** (exponent + 1) / (exponent + 1)
    return None


def _fast_antiderivative(
    integrand: sp.Basic,
    variable: sp.Symbol,
) -> sp.Basic | None:
    """Return a direct polynomial or elementary antiderivative when obvious."""
    if integrand.is_polynomial(variable):
        polynomial = sp.Poly(integrand, variable)
        terms: list[sp.Basic] = []
        for monomial, coefficient in polynomial.terms():
            exponent = monomial[0]
            terms.append(coefficient * variable ** (exponent + 1) / (exponent + 1))
        return sp.Add(*terms)
    coefficient, variable_part = integrand.as_independent(variable, as_Add=False)
    if coefficient.has(variable):
        return None
    direct = _fast_basic_antiderivative(variable_part, variable)
    if direct is None:
        return None
    return coefficient * direct


def _is_trigonometric_square(
    integrand: sp.Basic,
    variable: sp.Symbol,
) -> str | None:
    """Classify a sine or cosine square without simplifying unrelated expressions."""
    sine_square = sp.sin(variable) ** 2
    cosine_square = sp.cos(variable) ** 2
    if integrand == sine_square:
        return "sine"
    if integrand == cosine_square:
        return "cosine"
    # Equivalent forms such as 1 - cos(x)^2 are less common.  Only pay for
    # simplification when the expression can plausibly be such an identity.
    if integrand.func not in {sp.Add, sp.Pow}:
        return None
    if integrand.has(sp.sin) and sp.simplify(integrand - sine_square) == sp.Integer(0):
        return "sine"
    if integrand.has(sp.cos) and sp.simplify(integrand - cosine_square) == sp.Integer(0):
        return "cosine"
    return None


def _evaluate_indefinite_integral(
    integrand: sp.Basic,
    variable: sp.Basic,
) -> sp.Basic:
    integration_constant = sp.Symbol("C")
    if isinstance(variable, sp.Symbol):
        reciprocal_linear = match_reciprocal_linear_integral(integrand, variable)
        if reciprocal_linear is not None:
            return reciprocal_linear.result
        fast = _fast_antiderivative(integrand, variable)
        if fast is not None:
            return sp.Add(fast, integration_constant, evaluate=False)
    antiderivative = sp.integrate(integrand, variable)
    if isinstance(variable, sp.Symbol):
        trigonometric_square = _is_trigonometric_square(integrand, variable)
        if trigonometric_square == "sine":
            return sp.Add(
                variable / 2,
                -sp.sin(2 * variable) / 4,
                integration_constant,
                evaluate=False,
            )
        if trigonometric_square == "cosine":
            return sp.Add(
                variable / 2,
                sp.sin(2 * variable) / 4,
                integration_constant,
                evaluate=False,
            )
        if integrand.is_rational_function(variable) and not antiderivative.has(sp.log):
            decomposition = sp.apart(integrand, variable)
            terms = tuple(decomposition.as_ordered_terms())
            if str(decomposition) != str(integrand) and len(terms) > 1:
                return sp.Add(
                    *(sp.integrate(term, variable) for term in terms),
                    integration_constant,
                    evaluate=False,
                )
    return antiderivative + integration_constant


def _single_polynomial_target(
    equations: sp.Basic | Sequence[sp.Basic],
    symbols: tuple[sp.Basic, ...],
) -> tuple[sp.Equality, sp.Symbol] | None:
    """Return a lone equation that is polynomial in its only unknown, if it is one."""
    if len(symbols) != 1 or not isinstance(equations, sp.Equality):
        return None
    variable = symbols[0]
    if not isinstance(variable, sp.Symbol):
        return None
    difference = equations.lhs - equations.rhs
    if difference.free_symbols != {variable} or not difference.is_polynomial(variable):
        return None
    return equations, variable


def _solve_polynomial_over_complex_numbers(
    equation: sp.Equality,
    variable: sp.Symbol,
) -> list[dict[sp.Basic, sp.Basic]]:
    """Solve without the real-only assumption, then restore the caller's symbol."""
    complex_variable = sp.Symbol(variable.name)
    raw_solutions = sp.solve(
        equation.xreplace({variable: complex_variable}), complex_variable, dict=True
    )
    if not is_object_sequence(raw_solutions):
        return []
    solutions: list[dict[sp.Basic, sp.Basic]] = []
    for solution in raw_solutions:
        if not is_object_mapping(solution):
            return []
        root = solution.get(complex_variable)
        if not isinstance(root, sp.Basic):
            return []
        solutions.append({variable: root})
    return solutions


def _solution_is_real(solution: object) -> bool:
    """Return whether every value in a solved mapping is real-valued."""
    if not is_object_mapping(solution):
        return False
    return all(
        not isinstance(value, sp.Basic) or is_real_expression(value) for value in solution.values()
    )


def _solve_equations(
    equations: sp.Basic | Sequence[sp.Basic],
    symbols: tuple[sp.Basic, ...],
    domain: SolutionDomain | None = None,
) -> object:
    """Solve equations, respecting the requested ambient domain."""
    polynomial_target = _single_polynomial_target(equations, symbols)
    if polynomial_target is not None:
        solutions = _solve_polynomial_over_complex_numbers(*polynomial_target)
        if domain is SolutionDomain.REAL:
            return [solution for solution in solutions if _solution_is_real(solution)]
        return solutions
    raw_solutions = sp.solve(equations, *symbols, dict=True)
    if not is_object_sequence(raw_solutions):
        return []
    filtered: list[object] = []
    for solution in raw_solutions:
        if not is_object_mapping(solution):
            return []
        if domain is SolutionDomain.COMPLEX or _solution_is_real(solution):
            filtered.append(solution)
    return filtered


def _evaluate_definite_integral(
    integrand: sp.Basic,
    variable: sp.Basic,
    lower: sp.Basic,
    upper: sp.Basic,
) -> sp.Basic:
    """Evaluate a definite integral, including independently verified special families."""
    if isinstance(variable, sp.Symbol):
        sine_square_root = match_sine_square_root_integral(
            integrand,
            variable,
            lower,
            upper,
        )
        if sine_square_root is not None:
            return sine_square_root.value
    return sp.integrate(integrand, (variable, lower, upper))


class SympyExecutor:
    """Execute validated queries without handling presentation or pedagogy."""

    def __init__(self, converter: SympyConverter) -> None:
        """Use the supplied converter for query arguments and structured values."""
        self._converter = converter

    def _evaluate_sum(self, query: Query) -> sp.Basic:
        """Evaluate a sum, including exact series families SymPy leaves inert."""
        expect_arity(query, 4)
        expression = self._converter.to_sympy(query.arguments[0])
        variable = self._converter.to_sympy(query.arguments[1])
        lower = self._converter.to_sympy(query.arguments[2])
        upper = self._converter.to_sympy(query.arguments[3])
        if isinstance(variable, sp.Symbol):
            harmonic_sine = match_harmonic_sine_series(
                expression,
                variable,
                lower,
                upper,
            )
            if harmonic_sine is not None:
                return harmonic_sine.value
            alternating = match_alternating_p_series(expression, variable, lower, upper)
            if alternating is not None:
                return alternating.value
        return sp.summation(expression, (variable, lower, upper))

    def execute(self, query: Query) -> object:
        """Execute one validated query and return its backend-native result."""
        operation = query.operation
        arguments = query.arguments
        match operation:
            case Operation.SIMPLIFY:
                expect_arity(query, 1)
                return _simplify_expression(self._converter.to_sympy(arguments[0]))
            case Operation.EXPAND:
                expect_arity(query, 1)
                return sp.expand(self._converter.to_sympy(arguments[0]))
            case Operation.FACTOR:
                expect_arity(query, 1)
                return sp.factor(self._converter.to_sympy(arguments[0]))
            case Operation.CANCEL:
                expect_arity(query, 1)
                return sp.cancel(self._converter.to_sympy(arguments[0]))
            case Operation.APART:
                expect_arity(query, 1, 2)
                variable = self._converter.to_sympy(arguments[1]) if len(arguments) == 2 else None
                return sp.apart(self._converter.to_sympy(arguments[0]), variable)
            case Operation.SOLVE:
                expect_arity(query, 2)
                equations = self._converter.equations(arguments[0])
                symbols = self._converter.symbols(arguments[1])
                return _solve_equations(equations, symbols, query.domain)
            case Operation.NSOLVE:
                raise BackendError("numerical solves require the dedicated execution path")
            case Operation.SOLVE_INEQUALITY:
                expect_arity(query, 2)
                symbol = expect_symbol(arguments[1], role="inequality variable")
                backend_symbol = self._converter.to_sympy(symbol)
                if not isinstance(backend_symbol, sp.Symbol):
                    raise BackendError("symbol conversion did not produce a SymPy Symbol")
                return sp.solve_univariate_inequality(
                    self._converter.to_sympy(arguments[0]), backend_symbol
                )
            case Operation.DIFFERENTIATE:
                expect_arity(query, 2, 3)
                order = (
                    expect_integer(arguments[2], role="derivative order")
                    if len(arguments) == 3
                    else 1
                )
                return sp.diff(
                    self._converter.to_sympy(arguments[0]),
                    self._converter.to_sympy(arguments[1]),
                    order,
                )
            case Operation.INTEGRATE:
                expect_arity(query, 2, 4)
                integrand = self._converter.to_sympy(arguments[0])
                variable = self._converter.to_sympy(arguments[1])
                if len(arguments) == 2:
                    return _evaluate_indefinite_integral(integrand, variable)
                return _evaluate_definite_integral(
                    integrand,
                    variable,
                    self._converter.to_sympy(arguments[2]),
                    self._converter.to_sympy(arguments[3]),
                )
            case Operation.CONTOUR_INTEGRATE:
                raise BackendError("contour integrals require the dedicated execution path")
            case Operation.LIMIT:
                expect_arity(query, 3, 4)
                direction = "+-"
                if len(arguments) == 4:
                    direction_expression = arguments[3]
                    if not isinstance(direction_expression, Symbol):
                        raise QueryError("limit direction must be left, right, or two_sided")
                    direction = {"left": "-", "right": "+", "two_sided": "+-"}.get(
                        direction_expression.name,
                        "",
                    )
                    if not direction:
                        raise QueryError("limit direction must be left, right, or two_sided")
                return sp.limit(
                    self._converter.to_sympy(arguments[0]),
                    self._converter.to_sympy(arguments[1]),
                    self._converter.to_sympy(arguments[2]),
                    direction,
                )
            case Operation.SERIES:
                expect_arity(query, 4)
                return sp.series(
                    self._converter.to_sympy(arguments[0]),
                    self._converter.to_sympy(arguments[1]),
                    self._converter.to_sympy(arguments[2]),
                    expect_integer(arguments[3], role="series order"),
                )
            case Operation.SUM:
                return self._evaluate_sum(query)
            case Operation.PRODUCT:
                expect_arity(query, 4)
                return sp.product(
                    self._converter.to_sympy(arguments[0]),
                    (
                        self._converter.to_sympy(arguments[1]),
                        self._converter.to_sympy(arguments[2]),
                        self._converter.to_sympy(arguments[3]),
                    ),
                )
            case Operation.MATRIX:
                expect_arity(query, 1)
                return self._converter.matrix_from_expression(arguments[0])
            case Operation.DETERMINANT:
                expect_arity(query, 1)
                return self._converter.matrix_from_expression(arguments[0]).det()
            case Operation.INVERSE:
                expect_arity(query, 1)
                return self._converter.matrix_from_expression(arguments[0]).inv()
            case Operation.RANK:
                expect_arity(query, 1)
                return self._converter.matrix_from_expression(arguments[0]).rank()
            case Operation.RREF:
                expect_arity(query, 1)
                return self._converter.matrix_from_expression(arguments[0]).rref()
            case Operation.EIGENVALUES:
                expect_arity(query, 1)
                return self._converter.matrix_from_expression(arguments[0]).eigenvals()
            case Operation.DSOLVE:
                expect_arity(query, 2)
                return sp.dsolve(
                    self._converter.to_sympy(arguments[0]), self._converter.to_sympy(arguments[1])
                )
            case Operation.RSOLVE:
                expect_arity(query, 2)
                return sp.rsolve(
                    self._converter.to_sympy(arguments[0]), self._converter.to_sympy(arguments[1])
                )
            case Operation.LAPLACE:
                expect_arity(query, 3)
                return sp.laplace_transform(
                    self._converter.to_sympy(arguments[0]),
                    self._converter.to_sympy(arguments[1]),
                    self._converter.to_sympy(arguments[2]),
                    noconds=True,
                )
            case Operation.INVERSE_LAPLACE:
                expect_arity(query, 3)
                return sp.inverse_laplace_transform(
                    self._converter.to_sympy(arguments[0]),
                    self._converter.to_sympy(arguments[1]),
                    self._converter.to_sympy(arguments[2]),
                )
            case Operation.FOURIER:
                expect_arity(query, 3)
                return sp.fourier_transform(
                    self._converter.to_sympy(arguments[0]),
                    self._converter.to_sympy(arguments[1]),
                    self._converter.to_sympy(arguments[2]),
                )
            case Operation.INVERSE_FOURIER:
                expect_arity(query, 3)
                return sp.inverse_fourier_transform(
                    self._converter.to_sympy(arguments[0]),
                    self._converter.to_sympy(arguments[1]),
                    self._converter.to_sympy(arguments[2]),
                )
            case Operation.GCD:
                expect_arity(query, 2)
                return sp.gcd(
                    self._converter.to_sympy(arguments[0]), self._converter.to_sympy(arguments[1])
                )
            case Operation.LCM:
                expect_arity(query, 2)
                return sp.lcm(
                    self._converter.to_sympy(arguments[0]), self._converter.to_sympy(arguments[1])
                )
            case Operation.IS_PRIME:
                expect_arity(query, 1)
                return sp.isprime(expect_integer(arguments[0], role="primality input"))
            case Operation.PRIME_FACTORS:
                expect_arity(query, 1)
                return sp.factorint(expect_integer(arguments[0], role="factorization input"))
            case Operation.BINOMIAL | Operation.COMBINATIONS:
                expect_arity(query, 2)
                return sp.binomial(
                    self._converter.to_sympy(arguments[0]), self._converter.to_sympy(arguments[1])
                )
            case Operation.PERMUTATIONS:
                expect_arity(query, 2)
                n = self._converter.to_sympy(arguments[0])
                r = self._converter.to_sympy(arguments[1])
                return sp.factorial(n) / sp.factorial(n - r)
            case Operation.NUMERIC:
                expect_arity(query, 1, 2)
                digits = (
                    expect_integer(arguments[1], role="precision") if len(arguments) == 2 else 15
                )
                return sp.N(self._converter.to_sympy(arguments[0]), digits)
