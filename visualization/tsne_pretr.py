from tqdm import tqdm
import json
import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_labelled_set(stent_json, label = "malapposition", assert_nonempty = True):
    '''
    Checkout ivus-utils for a full set of json-parsing functions
    '''
    set4  = set()
    for n in range(len(stent_json['annotations'])):
        framelist = list(stent_json['annotations'][n]['frames'].keys()) # frames in the n-th annotation
        if stent_json['annotations'][n]['name'] == label:
            for f in framelist:
                if f not in set4:
                    set4.add(int(f))
    if assert_nonempty:
        if len(set4) == 0:
            raise ValueError(f"Empty set found for {label} label in json file")

    return set4

class PretrainedModel():
    def __init__(self, model, num_frames, transform):
        # Forward method is used for generating the embeddings
        self.model = model
        self.num_frames = num_frames
        self.transform = transform

    def video_embed(self, images):
        '''
        Embed videos sequentially to list of vectors
        '''

        embeddings = []
        for i in tqdm(range(len(images))):
            left_idx = i-self.num_frames//2
            right_idx = i-self.num_frames//2+self.num_frames
            clip = images[left_idx:right_idx]
            clip=self.transform(clip).to(next(self.model.parameters()).device)
            embedding = self.model.forward(clip).squeeze().to('cpu').detach().numpy()
            assert len(embedding.shape) == 1
            embeddings.append(embedding)

        return np.stack(embeddings)

def get_model(args):
    # HACK
    import sys

    print("Loading Saved Model")
    if args.model_type=="crw":
        path="/data/vision/polina/users/layjain/ivus-videowalk"
        sys.path.insert(0, path)
        import model_loader
        model, transform=model_loader.load_model(args.pretrained_path)
        model.forward = model.classifier_forward
        model = model.to(args.device)
        model.eval()
        model=PretrainedModel(model, 1, transform)
        sys.path.remove(path)
        return model

    elif args.model_type=="crw_classifier":
        path="/data/vision/polina/users/layjain/ivus-videowalk"
        sys.path.insert(0, path)
        import model_loader
        model, transform=model_loader.load_classifier(args.pretrained_path)
        print(f"Loaded Classifier \n {model}")
        model = model.to(args.device)
        model.eval()
        model=PretrainedModel(model, 1, transform)
        sys.path.remove(path)
        return model
    else:
        raise NotImplementedError


def get_labels(filename, num_frames, label_name="malapposition"):
    json_filepath = f"/data/vision/polina/users/layjain/pickled_data/malapposed_runs/jsons/{filename}.json"
    with open(json_filepath, "rb") as fh:
        stent_json = json.load(fh)
    
    labelled_indices = list(get_labelled_set(stent_json, label = label_name, assert_nonempty = True))
    ret = [0 for _ in range(num_frames)]
    for idx in labelled_indices:
        ret[idx]=1
    
    return ret

def main(args):
    root = '/data/vision/polina/users/layjain/pickled_data/malapposed_runs'
    model = get_model(args)
    for filepath in (glob.glob(os.path.join(root ,'*.pkl'))):
        filename = filepath.split('/')[-1].split('.')[0]
        print(f"Starting File {filename}")
        with open(filepath, 'rb') as fh:
                images = pickle.load(fh)
        assert isinstance(images, np.ndarray)
        images=np.squeeze(images)
        num_frames, H, W = images.shape
        assert H==W # square

        # Get the embeddings and labels
        model_embeddings = model.video_embed(images)
        for perplexity in [30]:
            tsne_embeddings = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=perplexity).fit_transform(model_embeddings)
            X = [e[0] for e in tsne_embeddings]
            Y = [e[1] for e in tsne_embeddings]
            if args.color_scheme=="labels":
                labels = get_labels(filename, len(model_embeddings))
                colors = ['red' if x==1 else 'black' for x in labels]
                plt.scatter(X, Y, c=colors, alpha=0.5); plt.title(filename)
            elif args.color_scheme=="index":
                colors = list(range(len(X))) # continuous gradient dark --> light
                # colors = [np.abs(x-len(labels)//2) for x in range(len(labels))]
                plt.scatter(X, Y, c=colors); plt.title(filename)
                plt.gray()
            elif args.color_scheme=="no-tsne":
                N, D = model_embeddings.shape
                assert D==2
                X = [e[0] for e in model_embeddings]; Y = [e[1] for e in model_embeddings]
                plt.scatter(X, Y, c=list(range(len(X))) ); plt.title(filename)

            if not os.path.exists(f"results/tsne/{args.model_name}"):
                os.makedirs(f"results/tsne/{args.model_name}")
            plt.savefig(f"results/tsne/{args.model_name}/{filename}_{perplexity}.jpg")
            plt.clf()
        break
        

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Label Propagation')

    parser.add_argument("--model-name",default="CLASSIFIER_36_F0",type=str)
    parser.add_argument("--model-type", default="crw_classifier", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--color-scheme", default="no-tsne", type=str)

    args=parser.parse_args()
    
    MODEL_PATHS={
        "PRETR_512_RRC" : "/data/vision/polina/users/layjain/ivus-videowalk/checkpoints/clean-sky-36/11-1-_len4-drop0-mlp1-lr0.01-temp0.007_len4-drop0-mlp1-lr0.01-temp0.007/model_149.pth",
        "CLASSIFIER_36_F0" : "/data/vision/polina/users/layjain/ivus-videowalk/checkpoints/12-13-36_lr_0.001-delta_100-model_crw-losswt_1.0-aug_['affine', 'flip', 'cj', 'blur', 'sharp']-sch_[100, 150]-nonorm_True/fold_0/model_63.pth"
    }

    args.pretrained_path = MODEL_PATHS[args.model_name]

    main(args)

