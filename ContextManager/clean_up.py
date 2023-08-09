import os

OUTPUT_PATHS = [
    "demo\\network_community_detection\\demo_output\\adv",
    "demo\\network_community_detection\\demo_output\\louvain",
]


class ContextManager:
    @staticmethod
    def delete_imgs() -> None:
        for path in OUTPUT_PATHS:
            for f_name in os.listdir(path):
                if f_name.endswith(".png"):
                    os.remove(os.path.join(f"{path}/", f_name))

    @staticmethod
    def delete_csvs() -> None:
        for path in OUTPUT_PATHS:
            path = f"{path}/csv_files"
            for f_name in os.listdir(path):
                if f_name.endswith(".csv"):
                    os.remove(os.path.join(f"{path}/", f_name))

    @staticmethod
    def clean_up() -> None:
        ContextManager.delete_imgs()
        ContextManager.delete_csvs()


ContextManager.clean_up()
