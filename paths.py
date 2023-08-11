OUTPUT_DIR = "demo"
CSV_DIR_SUFF = "csv_files"

IN_BRAIN_NETWORK_DIR = (
    "C:\\Users\\basia\\Desktop\\Praca_Inzynierska"
    + "\\QHyper\\demo\\network_community_detection\\brain_networks_data"
)
IN_BRAIN_NETWORK_FILE = "Edge_AAL90_Binary"

KARATE_PR_NAME = "karate"
BRAIN_PR_NAME = "brain"


def problem_dir(problem_name: str) -> str:
    return f"{OUTPUT_DIR}/{problem_name}"


# def resolution_dir()


def solver_dir(problem_name, solver_name: str) -> str:
    return f"{problem_dir(problem_name)}/{solver_name}"


def csv_path(problem_name: str, solver_name: str) -> str:
    return (
        f"{solver_dir(problem_name, solver_name)}/{CSV_DIR_SUFF}"
        + f"/{problem_name}_{solver_name}.csv"
    )


def img_dir(problem_name: str, solver_name: str) -> str:
    return (
        f"{solver_dir(problem_name, solver_name)}"
        + f"/{problem_name}_{solver_name}"
    )
