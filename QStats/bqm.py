problem = CommunityDetectionProblem(
    network_data=KarateClubNetwork(resolution=resolution), communities=1
)
binary_polynomial = dimod.BinaryPolynomial(
    problem.objective_function.as_dict(), dimod.BINARY
)
cqm = dimod.make_quadratic_cqm(binary_polynomial)
bqm, _ = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10)