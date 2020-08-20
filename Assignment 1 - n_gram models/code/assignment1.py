# Instructor: Karl Stratos
#
# Acknolwedgement: This exercise is heavily adapted from A1 of COS 484 at
# Princeton, designed by Danqi Chen and Karthik Narasimhan.

import argparse
import util
import matplotlib.pyplot as plt
import pdb
import numpy as np


def main(args):
    tokenizer = util.Tokenizer(tokenize_type=args.tok, lowercase=True)

    # TODO: you have to pass this test.
    util.test_ngram_counts(tokenizer)

    train_toks = tokenizer.tokenize(open(args.train_file).read())
    num_train_toks = int(args.train_fraction * len(train_toks))
    print('-' * 79)
    print('Using %d tokens for training (%g%% of %d)' %
          (num_train_toks, 100 * args.train_fraction, len(train_toks)))
    train_toks = train_toks[:int(args.train_fraction * len(train_toks))]
    val_toks = tokenizer.tokenize(open(args.val_file).read())

    train_ngram_counts = tokenizer.count_ngrams(train_toks)

    # Explore n-grams in the training corpus before preprocessing.
    util.show_ngram_information(train_ngram_counts, args.k,
                                args.figure_file, args.quiet)

    # Get vocab and threshold.
    print('Using vocab size %d (excluding UNK) (original %d)' %
          (min(args.vocab, len(train_ngram_counts[0])),
           len(train_ngram_counts[0])))
    vocab = [tup[0] for tup, _ in train_ngram_counts[0].most_common(args.vocab)]
    train_toks = tokenizer.threshold(train_toks, vocab, args.unk)
    val_toks = tokenizer.threshold(val_toks, vocab, args.unk)

    # # The language model assumes a thresholded vocab.
    lm = util.BigramLanguageModel(vocab, args.unk, args.smoothing,
                                  alpha=args.alpha, beta=args.beta)

    # Estimate parameters.
    lm.train(train_toks)

    train_ppl = lm.test(train_toks)
    val_ppl = lm.test(val_toks)
    print('Train perplexity: %f\nVal Perplexity: %f' %(train_ppl, val_ppl))

    # plot_laplace(args, vocab, train_toks, val_toks)
    # plot_interpolation(args, vocab, train_toks, val_toks)

    
# Question 5
def plot_laplace(args, vocab, train_toks, val_toks):
    alpha_ = [0.00001*pow(10,i) for i in range(7)]
    train_per = []
    val_per = []

    for i in range(7):
            # The language model assumes a thresholded vocab.
        print("--------- For alpha = ", alpha_[i], "----------")
        lm = util.BigramLanguageModel(vocab, args.unk, args.smoothing,
                                      alpha=alpha_[i], beta=args.beta)

        # Estimate parameters.
        lm.train(train_toks)

        train_ppl = lm.test(train_toks)
        val_ppl = lm.test(val_toks)
        print('Train perplexity: %f\nVal Perplexity: %f' %(train_ppl, val_ppl))
        train_per.append(train_ppl)
        val_per.append(val_ppl)

    # pdb.set_trace()
    plt.figure(figsize=(20,5))
    plt.close()
    plt.plot(np.arange(7), train_per, '-ro', label='Train Perplexity', linewidth=1)
    plt.plot(np.arange(7), val_per, '-bo', label='Val Perplexity', linewidth=1)
    plt.xticks(np.arange(7), labels = alpha_)
    plt.legend(loc="upper left")
    plt.xlabel('alpha')
    plt.show()
    plt.savefig('Laplace_train_val_perp.pdf', bbox_inches='tight')
    print('-' * 79)

# Question 7
def plot_interpolation(args, vocab, train_toks, val_toks):
    beta_ = np.arange(0.1,1,0.1)
    train_per = []
    val_per = []

    for i in range(9):
            # The language model assumes a thresholded vocab.
        print("--------- beta = ", beta_[i], "----------")
        lm = util.BigramLanguageModel(vocab, args.unk, args.smoothing,
                                      alpha=args.beta, beta=beta_[i])

        # Estimate parameters.
        lm.train(train_toks)

        train_ppl = lm.test(train_toks)
        val_ppl = lm.test(val_toks)
        print('Train perplexity: %f\nVal Perplexity: %f' %(train_ppl, val_ppl))
        train_per.append(train_ppl)
        val_per.append(val_ppl)

    # pdb.set_trace()
    # plt.figure(figsize=(20,5))
    plt.close()
    plt.plot(beta_, train_per, '-ro', label='Train Perplexity', linewidth=1)
    plt.plot(beta_, val_per, '-bo', label='Val Perplexity', linewidth=1)
    # plt.xticks(np.arange(7), labels = alpha_)
    plt.legend(loc="upper right")
    plt.xlabel('beta')
    plt.show()
    plt.savefig('interpolation.pdf', bbox_inches='tight')
    print('-' * 79)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str,
                        default='data/gigaword_subset.train',
                        help='corpus for training [%(default)s]')
    parser.add_argument('--val_file', type=str,
                        default='data/gigaword_subset.val',
                        help='corpus for validation [%(default)s]')
    parser.add_argument('--tok', type=str, default='nltk',
                        choices=['basic', 'nltk', 'wp', 'bpe'],
                        help='tokenizer type [%(default)s]')
    parser.add_argument('--vocab', type=int, default=10000,
                        help='max vocab size [%(default)d]')
    parser.add_argument('--k', type=int, default=10,
                        help='use top-k elements [%(default)d]')
    parser.add_argument('--train_fraction', type=float, default=1.0,
                        help='use this fraction of training data [%(default)g]')
    parser.add_argument('--smoothing', type=str, default=None,
                        choices=[None, 'laplace', 'interpolation'],
                        help='smoothing method [%(default)s]')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='parameter for Laplace smoothing [%(default)g]')
    parser.add_argument('--beta', type=float, default=0.8,
                        help='parameter for interpolation [%(default)g]')
    parser.add_argument('--figure_file', type=str, default='figure.pdf',
                        help='output figure file path [%(default)s]')
    parser.add_argument('--unk', type=str, default='<?>',
                        help='unknown token symbol [%(default)s]')
    parser.add_argument('--quiet', action='store_true',
                        help='skip printing n-grams?')
    args = parser.parse_args()
    main(args)
