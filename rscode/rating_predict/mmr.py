def MMR(itemScoreDict, similarityMatrix, lambdaConstant=0.5, topN=20):
    s, r = [], list(itemScoreDict.keys())
    while len(r) > 0:
        score = 0
        selectOne = None
        for i in r:
            firstPart = itemScoreDict[i]
            secondPart = 0
            for j in s:
                sim2 = similarityMatrix[i][j]
                if sim2 > secondPart:
                    secondPart = sim2
            equationScore = lambdaConstant * firstPart - (1 - lambdaConstant) * secondPart
            if equationScore > score:
                score = equationScore
                selectOne = i
        if selectOne == None:
            selectOne = i
        r.remove(selectOne)
        s.append(selectOne)
    return (s, s[:topN])[topN > len(s)]