# Enter your code here. Read input from STDIN. Print output to STDOUT

def read_words():
    with open('words.txt') as f:
        return [line.strip().lower() for line in f]
def strip_chars(word):
    word = word.lower().strip()
    if word.startswith('#'):
        return word[1:]
    if word.startswith('www'):
        word = word[4:]
    return word[:word.find('.')]
def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
def make_trie(words):
    root = dict()
    for word in words:
        current_dict = root
        for letter in word:
            current_dict = current_dict.setdefault(letter, {})
        current_dict = current_dict.setdefault(_end, _end)
    return root 
def in_trie(trie, word):
    current_dict = trie
    for letter in word:
        if letter in current_dict:
            current_dict = current_dict[letter]
        else:
            return False
    else:
        if _end in current_dict:
            return True
        else:
            return False
def possibles(s,index=0,token=''):
    if index >= len(s):
        return []
    token += s[index]
    index += 1
    while not in_trie(trie,token) and index < len(s):
        token += s[index]
        index += 1
    if not in_trie(trie,token):
        return []
    return [token] + possibles(s,index,token)
def tokenize(s):
    poss = possibles(s)
    if not poss:
        return []
    largest = poss[-1]
    poss.pop()
    inp = s[len(largest):]
    if len(inp) == 0:
        return [largest]
    tokenized = tokenize(inp)
    while not tokenized and len(poss) > 0:
        largest = poss[-1]
        poss.pop()
        inp = s[len(largest):]
        tokenized = tokenize(inp)
    if not tokenized:
        return []
    return [largest] + tokenized
        
#words = None
#read_words()
_end = '__end__'
words = read_words()
trie = make_trie(words)
#print tokenize('artisteer')
#exit()
n = input('How many examples ? ')
for x in xrange(n):
    word = strip_chars(raw_input('EnterHashtag:'))
    print ' '.join(tokenize(word))
