import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluation pipeline on MNIST')
    parser.add_argument('NoiseType', type=str, nargs='+',
                        help='type of noise to apply to the evaluation (default: ["White","Gaussian","SaltPepper"])')
    parser.add_argument('PathVAE', type=str, help='Path to the VAE/iVAE model')
    parser.add_argument('PathClassifier', type=str, help='Path to the classifier model')

    args = parser.parse_args()
    print(args.NoiseType)
    print(args.PathVAE)
    print(args.PathClassifier)

if __name__ == '__main__':
    main()