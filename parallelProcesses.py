import math

def vectorsPreProcessing(terms,arr,idxSet):
    results = []
    total = len(terms) * 1.0
    print('batch terms',terms)
    for i,term in enumerate(terms):
        # print(i,term)
        colIdx = i+1
        filter_class_spam = arr[:,0] == 1
        filter_class_legit = arr[:,0] == 0

        filter_term_exist = arr[:,colIdx] > 0
        filter_term_missing = arr[:,colIdx] == 0

        N11 = arr[filter_class_spam & filter_term_exist].shape[0]
        # print(N11)
        N10 = arr[filter_class_spam & filter_term_missing].shape[0]
        N01 = arr[filter_class_legit & filter_term_exist].shape[0]
        N00 = arr[filter_class_legit & filter_term_missing].shape[0]
        N = N11 + N10 + N01 +N00
        #N1. = N1X
        N1X = arr[filter_class_spam].shape[0]
        N0X = arr[filter_class_legit].shape[0]
        NX1 = arr[filter_term_exist].shape[0]
        NX0 = arr[filter_term_missing].shape[0]
        
        # print("term :", term, ", I(value) :", i)
        
        results.append({
            'colIdx':idxSet[i],
            'N11':N11,
            'N10':N10,
            'N01':N01,
            'N00':N00,
            'N1X':N1X,
            'N0X':N0X,
            'NX1':NX1,
            'NX0':NX0,
            'N':N
        })

    #     sys.stdout.write(f'\rterm #{colIdx}:[{term}] done. [{(i/total):0.2f}]%')
    # sys.stdout.flush()
    return results

def calculateMutualInfo(cols):
    results = []
    for col in cols:
        # Extract Info
        colIdx = col['colIdx']
        N11 = col['N11']
        N10 = col['N10']
        N01 = col['N01']
        N00 = col['N00']
        N1X = col['N1X']
        N0X = col['N0X']
        NX1 = col['NX1']
        NX0 = col['NX0']
        N = col['N']

        #calculate I
        I = (((N11/N)*math.log2(((N*N11)+1)/((N1X*NX1)+1))) + ((N01/N)*math.log2(((N*N01)+1)/((N0X*NX1)+1))) + 
            ((N10/N)*math.log2(((N*N10)+1)/((N1X*NX0)+1))) + ((N00/N)*math.log2(((N*N00)+1)/((N0X*NX0)+1))))

        results.append((colIdx,I))
    
    return results