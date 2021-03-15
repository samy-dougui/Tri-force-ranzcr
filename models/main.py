import argparse


from utils import main_predict, main_train, get_config

cfg = get_config()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type=str,
        choices=['resnet', 'efficientnet', 'ensemble'],
        help='Which model you want to use',
        required=True
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'test'],
        help='Pick which mode you want to run this file (train or test)',
        action='store',
        required=True
    )

    parser.add_argument(
        '--verbose',
        help='Will display model architecture and other logs',
        action='store_true'
    )

    args = parser.parse_args()
    if args.verbose:
        verbose = True
    else:
        verbose = False

    if args.mode == "train":
        if args.model == 'ensemble':
            print("Not implemented")
        else:
            main_train(cfg=cfg, model_name=args.model, verbose=verbose)
    elif args.mode == "test":
        main_predict(cfg=cfg, model_name=args.model, verbose=verbose)
    else:
        print("Unknown mode")
