import json
import numpy as np
import tqdm
from ausil import AuSiL
from tensorflow.keras.utils import Sequence, OrderedEnqueuer
import argparse


class FeaturesGenerator(Sequence):

    def __init__(self, txt_file):
        super(FeaturesGenerator, self).__init__()

        with open(txt_file) as f:
            videos = f.readlines()
        self.videos = [x.strip() for x in videos]

        self.indices = range(len(videos))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        name = self.videos[index].split()[0]
        try:
            wlafeatures = np.load(self.videos[index].split()[1])['features']
        except Exception as e:
            print(e)
            wlafeatures = np.array([])

        return name, wlafeatures


def load_queries(txt_file):

    print("loading queries....")

    queries_generator = FeaturesGenerator(txt_file)
    print(len(queries_generator))
    enqueuer = OrderedEnqueuer(queries_generator, use_multiprocessing=False, shuffle=False)
    enqueuer.start(workers=8, max_queue_size=32)
    generator = enqueuer.get()

    queries, querynames = [], []
    pbar = tqdm.trange(len(queries_generator))
    for _ in pbar:
        query_id, wlafeatures = next(generator)
        queries.append(wlafeatures)
        querynames.append(query_id)

    return queries, querynames


def calculate_similarities(db_file, querynames):

    feat_generator = FeaturesGenerator(db_file)
    enqueuer = OrderedEnqueuer(feat_generator, use_multiprocessing=True, shuffle=False)
    enqueuer.start(workers=8, max_queue_size=32)
    generator = enqueuer.get()

    similarities = dict({query: dict() for query in querynames})

    p_bar = tqdm.trange(len(feat_generator))
    for _ in p_bar:
        video, features = next(generator)
        if features.shape[0] == 0:
            continue
        # IF FRAMES < 10, CONCATENATE WITH ITSELF SO FRAMES >= 10
        while features.shape[0] < 10:
            features = np.concatenate((features, features), axis=0)

        sims = model.calculate_sim(features)

        for i, s in enumerate(sims):
            similarities[querynames[i]][video] = float(s)

    return similarities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--queries_file', type=str, help='queries txt file', required=True)
    parser.add_argument('-d', '--database_file', type=str, help='database txt file', required=True)
    parser.add_argument('-m', '--model_dir', help='trained model of AuSiL', required=True)
    parser.add_argument('-o', '--output_file', default='similarities.json', type=str, help='output file')
    parser.add_argument('-l', '--load_queries', action='store_true', help='Load queries to GPU memory')
    args = parser.parse_args()

    # Load queries from disk
    queries, querynames = load_queries(args.queries_file)
    # Initialize model
    model_path = args.model_dir + '/model'
    model = AuSiL(model_path, load_queries=args.load_queries,  gpu_id=0, queries_number=len(queries) if args.load_queries else None)

    # Extract query features and set them to GPU memory.
    print("Extracting queries features.....")
    features = []
    for query in queries:
        features.append(model.get_features(query))
    
    print("Setting queries to GPU.....")
    model.set_queries(features)

    # Calculate and store similarities
    print("Calculating Similarities.....")
    similarities = calculate_similarities(args.database_file, querynames)

    print('\nSaving Similarities in ' + args.output_file + '...')
    with open(args.output_file, 'w') as f:
        json.dump(similarities, f, indent=1)
