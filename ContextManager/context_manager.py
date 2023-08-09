import os

from paths import BRAIN_PR_NAME, CSV_DIR_SUFF, KARATE_PR_NAME, OUTPUT_DIR
from QStats.solvers.advantage.advantage import Advantage
from QStats.solvers.louvain.louvain import Louvain

SOLV_NAMES = [Advantage.name, Louvain.name]
PR_NAMES = [BRAIN_PR_NAME, KARATE_PR_NAME]
OUTPUT_PATHS = [
    f"{OUTPUT_DIR}/{p_name}/{s_name}"
    for s_name in SOLV_NAMES
    for p_name in PR_NAMES
]

IMG_SUFF = ".png"
CSV_SUFF = ".csv"


class ContextManager:
    @staticmethod
    def delete_imgs() -> list[str]:
        empty_paths = []
        for path in OUTPUT_PATHS:
            try:
                with os.scandir(path) as it:
                    for entry in it:
                        if entry.name.endswith(IMG_SUFF):
                            os.remove(os.path.join(f"{path}/", entry.name))
            except FileNotFoundError:
                continue
            finally:
                empty_paths.append(path)
        return empty_paths

    @staticmethod
    def delete_csvs() -> list[str]:
        empty_paths = []
        for path in OUTPUT_PATHS:
            path = f"{path}/{CSV_DIR_SUFF}"
            try:
                with os.scandir(path) as it:
                    for entry in it:
                        if entry.name.endswith(CSV_SUFF):
                            os.remove(os.path.join(f"{path}/", entry.name))
            except FileNotFoundError:
                continue
            finally:
                empty_paths.append(path)
        return empty_paths

    @staticmethod
    def rm_empty_dirs(dirs: list[str] | str) -> None:
        for path in dirs:
            try:
                os.removedirs(path)
            except OSError:
                continue

    @staticmethod
    def clean_up() -> None:
        empty_dirs = (
            ContextManager.delete_imgs() + ContextManager.delete_csvs()
        )
        ContextManager.rm_empty_dirs(empty_dirs)
