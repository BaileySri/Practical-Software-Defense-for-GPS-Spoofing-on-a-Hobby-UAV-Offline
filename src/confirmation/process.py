import pandas as pd
from functools import reduce
#TQDM for progress information
from tqdm.notebook import trange

#Open log file and parse out the SNS data into pandas dataframes
def process_SNS(path, out=""):
    SNSs = [[], [], []]
    with open(path) as infile:
        number_lines = sum(1 for line in open(path))
        for counter in trange(number_lines):
            line = infile.readline()
            strippedLine = line.strip()
            if "FMT" in strippedLine:
                if "SNS1" in strippedLine:
                    SNSs[0] = pd.DataFrame(columns=strippedLine.split(',')[5:])
                if "SNS2" in strippedLine:
                    SNSs[1] = pd.DataFrame(columns=strippedLine.split(',')[5:])
                if "SNS3" in strippedLine:
                    SNSs[2] = pd.DataFrame(columns=strippedLine.split(',')[5:])
            else:
                if "SNS1" in strippedLine:
                    SNSs[0].loc[len(SNSs[0])] = strippedLine.split(',')[1:]
                if "SNS2" in strippedLine:
                    SNSs[1].loc[len(SNSs[1])] = strippedLine.split(',')[1:]
                if "SNS3" in strippedLine:
                    SNSs[2].loc[len(SNSs[2])] = strippedLine.split(',')[1:]
    SNS = reduce(lambda left, right: pd.merge(left, right, on=' TimeUS'), SNSs)
    if out:
        SNS.to_csv(out, index=False)
    return(SNS)