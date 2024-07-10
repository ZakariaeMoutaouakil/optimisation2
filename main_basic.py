from argparse import ArgumentParser
from ast import literal_eval
from json import dumps
from time import time
from typing import List, Tuple, Dict

from pandas import read_csv
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from certify.smallest_subset import smallest_subset
from experiment.find_q_for_quantile_exact import find_q_for_quantile_exact
from utils.logging_config import setup_logger
from utils.remove_zeros import remove_zeros

parser = ArgumentParser(description='Certify many examples')
parser.add_argument("--data", type=str, help="Location of tsv data", required=True)
parser.add_argument("--outfile", type=str, help="Location of output tsv file", required=True)
parser.add_argument("--log", type=str, help="Location of log file", required=True)
parser.add_argument("--m", type=int, default=3, help="Number of coordinates")
parser.add_argument("--alpha", type=float, default=0.01, help="Failure probability")
parser.add_argument("--sigma", type=float, default=0.12, help="Noise parameter")
parser.add_argument("--num_simulations", type=int, default=100000, help="Number of simulations")
args = parser.parse_args()

logger = setup_logger("certification", args.log)

# Use pprint to log the arguments in a more readable format
logger.info("Parsed arguments:")
args_dict = vars(args)

# Pretty print the dictionary with json.dumps
formatted_args = dumps(args_dict, indent=4)

# Log the formatted arguments
logger.info(formatted_args)

# Define the data types for each column
dtype_dict = {
    'idx': int,
    'label': int,
    'predict': int,
    'radius': float,
    'correct': int,
    'time': str
}

# Read the preprocessed TSV data into a DataFrame with specified dtypes
df = read_csv(args.data, sep='\t', dtype=dtype_dict, converters={'counts': literal_eval})

n = sum(df.iloc[0]['counts'])
logger.info("n: " + str(n))

# Dictionary to cache results and time of final_result function
final_result_cache: Dict[Tuple[int, ...], Tuple[float, float]] = {}
elapsed_time, cached_time = 0., 0.

for i in range(len(df)):
    logger.info("old:")
    logger.info(df.iloc[i])
    counts: List[int] = df.iloc[i]['counts']
    prediction = counts.index(max(counts))
    reduced_counts = remove_zeros(coords=tuple(counts), min_dimension=args.m)
    logger.debug("reduced_counts: " + str(reduced_counts))
    observation = sorted(smallest_subset(vector=reduced_counts, num_partitions=args.m))
    reduced_counts_tuple = tuple(int(x) for x in observation)
    logger.debug("reduced_counts_tuple: " + str(reduced_counts_tuple))
    radius = 0.

    if reduced_counts_tuple in final_result_cache:
        radius, cached_time = final_result_cache[reduced_counts_tuple]
    else:
        start_time = time()
        p1 = find_q_for_quantile_exact(n=n, m=args.m, alpha=args.alpha, x=max(reduced_counts_tuple),
                                       num_simulations=args.num_simulations)
        end_time = time()
        elapsed_time = end_time - start_time
        logger.info("elapsed_time: " + str(elapsed_time))
        logger.debug("p1: " + str(p1))
        p1_ = proportion_confint(max(reduced_counts_tuple), n, alpha=2 * args.alpha, method="beta")[0]
        logger.debug("p1_: " + str(p1_))

        if p1 > 0.5:
            radius = args.sigma * norm.ppf(p1)
            logger.debug("radius: " + str(radius))
        else:
            prediction = -1
            logger.warning("Don't certify this example")

        # Cache the result and time
        final_result_cache[reduced_counts_tuple] = (radius, elapsed_time)

    logger.debug("radius: " + str(radius))
    logger.debug("prediction: " + str(prediction))
    df.loc[df.index[i], 'radius'] = radius
    df.loc[df.index[i], 'correct'] = int(prediction == df.iloc[i]['label'])
    df.loc[df.index[i], 'predict'] = prediction

    if reduced_counts_tuple not in final_result_cache:
        df.loc[df.index[i], 'time'] = f"{elapsed_time:.6f}"
    else:
        df.loc[df.index[i], 'time'] = f"{cached_time:.6f}"
    logger.info("new:")
    logger.info(df.iloc[i])

# Remove the last three columns
df_modified = df.iloc[:, :-2]  # This slices out all rows and all columns except the last two

# Save to TSV file
df_modified.to_csv(args.outfile, sep='\t',
                   index=False)  # Set index=False if you don't want to save the index as a separate column
