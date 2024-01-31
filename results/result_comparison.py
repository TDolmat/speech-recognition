import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


results = pd.read_csv("results/result_comparison.csv", sep='\t')

def average_of_row(rows, row_name):
    total_count = 0
    rows_len = len(rows)

    for i in range(rows_len):
        total_count += float(rows.iloc[i][row_name])
    return total_count/rows_len

def print_test_results():
    print("GREEDY")
    print(f"Average WER: {average_of_row(results, 'greedy_wer')}")
    print(f"Average CER: {average_of_row(results, 'greedy_cer')}")
    print(f"WER std: {np.std(results.greedy_wer)}")
    print(f"CER std: {np.std(results.greedy_cer)}")

    print("\nBEAM")
    print(f"Average WER: {average_of_row(results, 'beam_search_wer')}")
    print(f"Average CER: {average_of_row(results, 'beam_search_cer')}")
    print(f"WER std: {np.std(results.beam_search_wer)}")
    print(f"CER std: {np.std(results.beam_search_cer)}")


def plot_greedy_wer_histogram():
    plt.figure(figsize=(8,6))

    plt.hist(results.greedy_wer, bins=20, color="royalblue", edgecolor='white')
    plt.xlabel('WER Values')
    plt.ylabel('Frequency')
    plt.title('Disribution of WER in results of raw acoustic model')

    plt.show()
 
def plot_greedy_cer_histogram():
    plt.figure(figsize=(8,6))

    plt.hist(results.greedy_cer, bins=20, color="skyblue", edgecolor='white')
    plt.xlabel('CER Values')
    plt.ylabel('Frequency')
    plt.title('Disribution of CER in results of raw acoustic model')

    plt.show()

def plot_beam_wer_histogram():
    plt.figure(figsize=(8,6))

    plt.hist(results.beam_search_wer, bins=20, color="orange", edgecolor='white')
    plt.xlabel('WER Values')
    plt.ylabel('Frequency')
    plt.title('Disribution of WER in results of model with KenLM and CTC beam search')

    plt.show()

def plot_beam_cer_histogram():
    plt.figure(figsize=(8,6))

    plt.hist(results.beam_search_cer, bins=20, color="peachpuff", edgecolor='white')
    plt.xlabel('CER Values')
    plt.ylabel('Frequency')
    plt.title('Disribution of CER in results of model with KenLM and CTC beam search')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', default=False, required=False, action='store_true',
                        help='print test results')
    parser.add_argument('-gw', default=False, required=False, action='store_true',
                        help='show greedy WER histogram')
    parser.add_argument('-gc', default=False, required=False, action='store_true',
                        help='show greedy CER histogram')
    parser.add_argument('-bw', default=False, required=False, action='store_true',
                        help='show beam search WER histogram')
    parser.add_argument('-bc', default=False, required=False, action='store_true',
                        help='show beam search CER histogram')

    args = parser.parse_args()

    if args.p:
        print_test_results()
    elif args.gw:
        plot_greedy_wer_histogram()
    elif args.gc:
        plot_greedy_cer_histogram()
    elif args.bw:
        plot_beam_wer_histogram()
    elif args.bc:
        plot_beam_cer_histogram()