
import os
import pandas as pd
import numpy as np
import requests
import time
import re

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------
def request_with_retries(url, max_retry, retry=0):
    '''returns a http reponse to the request to target url. If '''
    resp = requests.get(url)
    if resp.ok:
        return resp
    elif retry < max_retry:
        time.sleep(2**retry)
        return request_with_retries(url, max_retry, retry=(retry + 1))
    else:
        return resp

def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    req = request_with_retries(url, 2)
    text = req.text
    text = re.sub('\r\n', '\n', text)
    start_end = re.findall('\*\*\*[A-Z0-9 ]+\*\*\*', text)
    start = text.find(start_end[0])
    end = text.find(start_end[1])
    content = text[start + len(start_end[0]):end]
    content

    return content
    
# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of any paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of any paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens in the sequence of words are split 
        apart at 'word boundaries' (see the regex lecture).
        - Tokens should include no whitespace.

    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """
    text2 = re.sub('(\n\n)+', '\x03\x02', book_string)
    split = re.findall(r"[A-Za-z0-9]+|[-_().,!?;:']|[\x03\x02]", text2)
    if split[-1] == '\x02':
        split = split[:-1]
    elif split[-1] != '\x03':
        split = split + ['\x03']
    if split[0] == '\x03':
        split == split[1]
    elif split[0] != '\x02':
        split = ['\x02'] + split

    return split
    
# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        to_series = pd.Series(1 / len(np.array(np.unique(tokens))), index=np.array(np.unique(tokens)))
        return to_series
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        prob = 1
        for i in words:
            if i not in self.mdl.index:
                prob = prob * 0
            else:
                prob_item = self.mdl.get(i)
                prob = float(prob) * prob_item
        return prob
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        result = ''
        for i in range(M):
            result += self.mdl.sample(n=1).index[0] + ' '
        return result

            
# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        lst = list(tokens)
        index = list(set(tokens))
        value = []
        for i in index:
            value.append(lst.count(i)/len(lst))
        ser = pd.Series(value, index=index)
        return ser
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        prob = 1
        for i in words:
            if i not in self.mdl.index:
                prob = prob * 0
            else:
                prob_item = self.mdl.get(i)
                prob = float(prob) * prob_item
        return prob
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        result = ''
        for i in range(M):
            result += self.mdl.sample(n=1, weights=self.mdl).index[0] + ' '
        return result
        
    
# ---------------------------------------------------------------------
# Question #5,6,7,8
# ---------------------------------------------------------------------

class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """

        self.N = N
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            mdl = NGramLM(N-1, tokens)
            self.prev_mdl = mdl

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        result = []
        lst = list(tokens)
        for i in range(len(lst) - self.N + 1):
            nlist = [lst[j] for j in range(i, i + self.N)]
            result.append(tuple(nlist))
        return result
        
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe with three columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """

        # ngram counts C(w_1, ..., w_n)
        counts = np.zeros(len(ngrams))
        for i in range(len(ngrams)):
            counts[i] = ngrams.count(ngrams[i])

        # n-1 gram counts C(w_1, ..., w_(n-1))
        precede = list(map(lambda x: x[:-1], ngrams))
        counts2 = np.zeros(len(precede))
        for i in range(len(precede)):
            counts2[i] = precede.count(precede[i])
        cond_prob = counts / counts2

        result = pd.DataFrame()
        result['ngram'] = ngrams
        result['n1gram'] = precede
        result['prob'] = cond_prob

        return result.drop_duplicates(subset='ngram').reset_index(drop=True)
        # return result

        # Create the conditional probabilities

        # Put it all together
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4)*(1/2)*(1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """

        prob = 1

        base_case = []
        for i in range(len(self.ngrams)):
            if i == 0:
                base_case += list(self.ngrams[i])
            else:
                for j in list(self.ngrams[i])[self.N - 1:]:
                    base_case.append(j)

        base_case = pd.Series(base_case)
        base_prob = base_case.value_counts(normalize=True)

        for i in words:
            if i not in base_case.to_list():
                prob *= 0

        prob = prob * base_prob.get(words[0])

        '''
        subtoken = []
        for i in range(len(words)-self.N+1):
            nlist = [words[j] for j in range(i, i+self.N)]
            subtoken.append(tuple(nlist))
        for i in subtoken:
            if i not in self.mdl['ngram'].to_list():
                prob = prob*0
        '''

        '''
        for i in words:
            for j in self.mdl['ngram']:
                if (words.index(i) == 0 and j[0] == i):
                    words.index(i)!=0 and j[1] == i:
                    prob = prob * self.mdl.loc[self.mdl['ngram']==j]['prob'].values[0]
        '''
        n1 = self.mdl
        n = self.N
        new_ngrams = self.create_ngrams(words)
        cond_prob = 1

        for i in new_ngrams:
            for j in n1['ngram'].to_list():
                if j == i:
                    prob = prob * n1.loc[n1['ngram'] == j]['prob'].values[0]
        while n > 2:
            n1 = self.train(n1['n1gram'].to_list())
            new_ngrams1 = list(map(lambda x: x[:-1], new_ngrams))
            for i in new_ngrams1:
                for j in n1['ngram'].to_list():
                    if j == i:
                        prob = prob * n1.loc[n1['ngram'] == j]['prob'].values[0]
            n -= 1

        return prob

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """

        # Use a helper function to generate sample tokens of length `length`

        sample = []
        first = self.mdl.sample()["ngram"].tolist()[0]
        for i in first:
            sample.append(i)
        precede = first[1:]
        for i in range(M-self.N):
            table = self.mdl[self.mdl["n1gram"] == precede]
            df = table.copy()
            df["weight"] = df["ngram"].apply(lambda x: self.probability(x))
            if df.size == 0:
                gene = self.mdl.sample(weights=df["weight"])["ngram"].tolist()[0][0]
            else:
                gene = df.sample(weights=df["weight"])["ngram"].tolist()[0][0]
            sample.append(gene)
            precede = list(precede)[1:]
            precede.append(gene)
            precede = tuple(precede)
        
        # Transform the tokens to strings
        result = '\x02'
        for s in sample:
            result = result + " " + s
        return result


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_book'],
    'q02': ['tokenize'],
    'q03': ['UniformLM'],
    'q04': ['UnigramLM'],
    'q05': ['NGramLM']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
