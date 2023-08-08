import dimod
from QHyper.problems.base import Problem


class BQM:
    @staticmethod
    def bqm(problem: Problem, lagrange_multiplier: float):
        binary_polynomial = dimod.BinaryPolynomial(
            problem.objective_function.as_dict(), dimod.BINARY
        )
        cqm = dimod.make_quadratic_cqm(binary_polynomial)
        bqm, _ = dimod.cqm_to_bqm(cqm, lagrange_multiplier=lagrange_multiplier)
        return bqm
