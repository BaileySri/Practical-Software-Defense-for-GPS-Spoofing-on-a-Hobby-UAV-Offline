import pandas as pd
from functools import reduce
#TQDM for progress information
from tqdm.notebook import trange

#Open log file and parse out the SNS data into pandas dataframes
def process_SNS(path, out="", SNS_COUNT=4):
    SNSs = [{} for i in range(SNS_COUNT)]
    SNSs_counter = [0] * SNS_COUNT
    cols = [None] * SNS_COUNT

    with open(path) as infile:
        number_lines = sum(1 for line in open(path))
        for counter in trange(number_lines):
            line = infile.readline()
            strippedLine = line.strip()

            if "FMT" in strippedLine:
                for i in range(1, SNS_COUNT+1):
                    if ("SNS" + str(i)) in strippedLine:
                        cols[i-1] = [val.strip() for val in strippedLine.split(',')[5:]]
                        break
            else:
                for i in range(1, SNS_COUNT+1):
                    if ("SNS" + str(i)) in strippedLine:
                        SNSs[i-1][SNSs_counter[i-1]] = dict(zip(cols[i-1], [val.strip() for val in strippedLine.split(',')[1:]]))
                        SNSs_counter[i-1] = SNSs_counter[i-1] + 1
                        break
        
        for i in range(SNS_COUNT):
            SNSs[i] = pd.DataFrame.from_dict(SNSs[i], "index")

    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)

    if out:
        SNS.to_csv(out, index=False)
    return(SNS)