import os

IMG_PATHS = [
    "demo\\network_community_detection\\demo_output\\adv",
    "demo\\network_community_detection\\demo_output\\louvain",
]


class ContextManager:
    @staticmethod
    def delete_imgs() -> None:
        for path in IMG_PATHS:
            for f_name in os.listdir(path):
                if f_name.endswith(".png"):
                    os.remove(os.path.join(f"{path}/", f_name))
