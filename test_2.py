from typing import Callable, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.base import BaseEstimator
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from hydra_zen import builds, load_from_yaml, make_config, store, zen

# 1. Configuring multiple datasets and classifiers

###############
# Classifiers #
###############
# Created configurations for various classifiers and store
# the configs in hydra-zen's config store.

classifier_store = store(group="classifier")

classifier_store(KNeighborsClassifier, n_neighbors=3, name="knn")
classifier_store(SVC, kernel="linear", C=0.025, name="svc_linear")
classifier_store(SVC, gamma=2, C=1, name="svc_rbf")
classifier_store(
    GaussianProcessClassifier,
    kernel=builds(RBF, length_scale=1.0),
    name="gp",
)
classifier_store(DecisionTreeClassifier, max_depth=5, name="decision_tree")
classifier_store(
    RandomForestClassifier,
    max_depth=5,
    n_estimators=10,
    max_features=1,
    name="random_forest",
)
classifier_store(MLPClassifier, alpha=1, max_iter=1000, name="mlp")
classifier_store(AdaBoostClassifier, name="ada_boost")
classifier_store(GaussianNB, name="naive_bayes")
classifier_store(QuadraticDiscriminantAnalysis, name="qda")

############
# Datasets #
############
# Created configurations for various datasets and store
# the configs in hydra-zen's config store.

dataset_store = store(group="dataset")

# For the linear dataset, add a wrapper that
# randomly spaces our the data
def add_random_scattering(make_dataset):
    def wraps(*args, **kwargs):
        X, y = make_dataset(*args, **kwargs)
        rng = np.random.RandomState(2)
        X += 2 * rng.uniform(size=X.shape)
        return X, y
    return wraps


dataset_store(
    make_classification,
    zen_wrappers=add_random_scattering,  # <- apply wrapper here
    zen_partial=True,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
    name="linear",
)
dataset_store(
    make_moons,
    zen_partial=True,
    noise=0.3,
    random_state=0,
    name="moons",
)
dataset_store(
    make_circles,
    zen_partial=True,
    noise=0.2,
    factor=0.5,
    random_state=1,
    name="circles",
)



#####################################
# Configure and store task function #
#####################################

# Task configuration:
#    Set the default dataset to be `moons`
#    and the default classifier to be `knn`
store(
    make_config(
        hydra_defaults=["_self_" , {"dataset": "moons"}, {"classifier": "knn"}],
        dataset=None,
        classifier=None,
    ),
    name="config",
)

# 2. Build a task function to load data, fit a classifier, and plot the result.

def task(
    dataset: Callable[[], Tuple[np.ndarray, np.ndarray]],
    classifier: BaseEstimator,
):
    fig, ax = plt.subplots()

    assert isinstance(ax, Axes)
    assert isinstance(fig, Figure)

    # create and split dataset for train and test
    X, y = dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # plot the data
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    cm = plt.cm.RdBu  # type: ignore
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")  # type: ignore

    # Plot the testing points
    ax.scatter(
        X_test[:, 0],  # type: ignore
        X_test[:, 1],  # type: ignore
        c=y_test,
        cmap=cm_bright,
        alpha=0.6,
        edgecolors="k",
    )

    # Fit classifier on data
    clf = make_pipeline(StandardScaler(), classifier)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    DecisionBoundaryDisplay.from_estimator(clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_axis_off()
    ax.text(
        x_max - 0.3,
        y_min + 0.3,
        ("%.2f" % score).lstrip("0"),
        size=25,
        horizontalalignment="right",
    )

    # load overrides to set plot title
    overrides = load_from_yaml(".hydra/overrides.yaml")


    # 3. Save the figure in the local experiment directory.
    if len(overrides) == 2:
        # Running in multirun mode: save fig based
        # on dataset/classifier overrides
        dname = overrides[0].split("=")[1]
        cname = overrides[1].split("=")[1]
        fig.savefig(f"{dname}_{cname}.png", pad_inches=0.0, bbox_inches = 'tight')
    else:
        # Not guaranteed to have overrides, just save as result.png
        fig.savefig("result.png", pad_inches=0.0, bbox_inches = 'tight')

    # For hydra multirun figures will stay open until all runs are completed
    # if we do not close the figure
    plt.close()


if __name__ == "__main__":
    from hydra.conf import HydraConf, JobConf
    # Configure Hydra to change the working dir to match that of the output dir
    store(HydraConf(job=JobConf(chdir=True)), name="config", group="hydra")

    # Add all of the configs, that we put in hydra-zen's (local) config store,
    # to Hydra's (global) config store.
    store.add_to_hydra_store(overwrite_ok=True)

    # Use `zen()` to convert our Hydra-agnostic task function into one that is
    # compatible with Hydra.
    # Use `.hydra_main(...)` to generate the Hydra-compatible CLI for our program.
    zen(task).hydra_main(config_path=None, config_name="config", version_base="1.2")