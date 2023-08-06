from gpt2_embed2vec import sentiment_algebra
import sys

if len(sys.argv) < 4:
    missing = 4 - len(sys.argv)
    for i in range(missing):
        arg = input("Enter argument %d: " % (i + 1))
        sys.argv.append(arg)
else: 
    inp = sys.argv[1]
    sent1 = sys.argv[2]
    sent2  = sys.argv[3]


#while True:
#    command = input("Enter quit to exit, or ask a new question (var1 = arg1\n...): ")
#    
#    if command == "quit":
#        break

res = sentiment_algebra(inp,sent1,sent2)
print(res)
