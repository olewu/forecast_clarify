import wandb
import os


PROJECT = "climate-futures-clarify"
ENTITY_DEFAULT = "clarify"
ARTIFACT_DEFAULT = "sea_temperature:latest"
WANDB_TOKEN = os.getenv("WANDB_TOKEN")


class Manager(object):
    _instance = None

    def __new__(cls, project=PROJECT, entity=ENTITY_DEFAULT):
        if cls._instance is None:
            cls._instance = super(Manager, cls).__new__(cls)
            cls._downloaded = False
            cls.project = project
            cls.entity = entity
            cls.dir = "./data"
            wandb.login(key=WANDB_TOKEN)
        return cls._instance

    def download_datasets(cls, artifact=ARTIFACT_DEFAULT, force=False):
        if (not cls._downloaded) or force:
            print("Downloading datasets from model-registry...")
            run = wandb.init(project=cls.project, entity=cls.entity)
            datasets = run.use_artifact(artifact)
            cls.dir = datasets.download()
            cls._downloaded = True
            run.finish()
        else:
            print("Datasets already downloaded...")

    def get_dir(cls) -> str:
        cls.download_datasets()
        return cls.dir

    def push_datasets(cls, destination_path: str = "data"):
        run = wandb.init(project=cls.project, entity=cls.entity)
        artifact = wandb.Artifact("sea_temperature", type="dataset")
        artifact.add_dir(destination_path)
        run.log_artifact(artifact)

        run.finish()
