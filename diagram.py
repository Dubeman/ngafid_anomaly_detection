from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.aws.database import RDS
from diagrams.programming.language import Python
from diagrams.onprem.vcs import Git

with Diagram("Predictive Maintenance Architecture", show=False, filename="predictive_maintenance"):
    with Cluster("Data Sources"):
        kaggle_data = Custom("", "./images/kaggle.png")  # CSV data source represented by Kaggle logo

    with Cluster("Data Processing"):
        preprocessing = Python("")
        feature_engineering = Python("")

    with Cluster("Clustering and Modeling"):
        clustering = Python("")
        modeling = Python("")

    with Cluster("Storage"):
        feature_storage = RDS("ROCKET features")
        processed_data = RDS("Intermediate transfomed data")
        rc_cluster = Custom("", "./images/rc_cluster.png")  # RC Cluster represented by logo

    with Cluster("Analysis and Reporting"):
        jupyter = Custom("", "./images/jupyter.png")  # Jupyter Notebooks logo
        matplotlib = Custom("", "./images/matplotlib.png")  # Matplotlib logo
        mlflow = Custom("", "./images/mlflow.png")  # MLflow logo

    # GitHub for version control
    github = Git("")  # GitHub represented by its logo

    # Data flow arrows
    kaggle_data >> Edge(label="Load & Process") >> preprocessing
    preprocessing >> Edge(label="Generate Features") >> feature_engineering
    feature_engineering >> Edge(label="") >> feature_storage

    # Clustering and modeling flow
    feature_storage >> Edge(label="Cluster Data") >> clustering
    clustering >> Edge(label="Model Data") >> modeling
    # modeling >> Edge(label="Store Results") >> processed_data

    # Analysis and reporting
    processed_data >> Edge(label="Analysis") >> jupyter
    processed_data >> Edge(label="Visualization") >> matplotlib
    processed_data >> Edge(label="Tracking") >> mlflow

    # Version control integration
    for cluster in [("Data Sources", [kaggle_data]),
                    ("Data Processing", [preprocessing, feature_engineering]),
                    ("Clustering and Modeling", [clustering, modeling]),
                    ("Storage", [feature_storage, processed_data]),
                    ("Analysis and Reporting", [jupyter, matplotlib, mlflow])]:
        with Cluster(cluster[0]):
            cluster_elements = cluster[1]
            github_edge = Edge(label="")  # Common edge to GitHub
            cluster_elements[0] >> github_edge >> github
