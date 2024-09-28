import argparse


def args_parser():

    parser = argparse.ArgumentParser()
    # scenario ans federated learning
    parser.add_argument(
        "--epochs", type=int, default=3, help="number of rounds of training"
    )
    parser.add_argument(
        "--clients_num", type=int, default=15, help="number of clients: K"
    )
    parser.add_argument(
        "--clusters_num", type=int, default=None, help="number of clusters: C"
    )

    # workspace arguments
    parser.add_argument(
        "--clean_dataset",
        type=bool,
        default=True,
        help="clean\
                        the model/data_set or not",
    )
    parser.add_argument(
        "--clean_user",
        type=bool,
        default=True,
        help="clean\
                        the user/ or not",
    )
    parser.add_argument(
        "--clean_clients",
        type=bool,
        default=True,
        help="clean\
                        the model/clients or not",
    )

    # data set
    parser.add_argument("--dataset", type=str, default="TMDB", help="name of dataset")

    # other arguments
    parser.add_argument(
        "--gpu",
        default=None,
        help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="type \
                        of optimizer",
    )
    parser.add_argument("--verbose", type=int, default=1, help="verbose")
    parser.add_argument("--seed", type=int, default=1, help="random seed")

    args = parser.parse_args()
    return args
