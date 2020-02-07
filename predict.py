from arguments import PREDICT_ARGS_LIST, create_parser
from utils.network_utils import load_model, predict
from utils.data_utils import load_categories
import numpy as np

def main():
    parser = create_parser(description="Image Classifier", **PREDICT_ARGS_LIST)

    args = parser.parse_args()

    #print(args)
    #raise SystemExit

    model, _ = load_model(args.checkpoint)

    cats = load_categories(args.category_names)

    probs, classes = predict(model, args.input, args.top_k, args.device)

    labels = []
    for cl in classes:
        labels.append(cats[cl])
    
    for label, prob in zip(labels, probs):
        print(label, f"{prob * 100:.3f}%")

if __name__ == '__main__':
    main()