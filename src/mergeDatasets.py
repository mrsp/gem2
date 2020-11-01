import numpy as np
import glob


def main():


    #accX 1
    if(glob.glob('../GEM2_nao_training/**/accX.txt')):
        files = glob.glob('../GEM2_nao_training/**/accX.txt')
    else:
        print("No files in path")
        return -1

    with open( '../GEM2_nao_training/accX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #gt
    if(glob.glob('../GEM2_nao_training/**/gt.txt')):
        files = glob.glob('../GEM2_nao_training/**/gt.txt')
        with open( '../GEM2_nao_training/gt.txt', 'w' ) as result:
            for file_ in files:
                for line in open( file_, 'r' ):
                    result.write( line )
    else:
        print("No GT in path")


    #accY 2
    files = glob.glob('../GEM2_nao_training/**/accY.txt')
    with open( '../GEM2_nao_training/accY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #accZ 3
    files = glob.glob('../GEM2_nao_training/**/accZ.txt')
    with open( '../GEM2_nao_training/accZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    
    #baccX_LL 4
    files = glob.glob('../GEM2_nao_training/**/baccX_LL.txt')
    with open( '../GEM2_nao_training/baccX_LL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccY_LL 5 
    files = glob.glob('../GEM2_nao_training/**/baccY_LL.txt')
    with open( '../GEM2_nao_training/baccY_LL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccZ_LL 6
    files = glob.glob('../GEM2_nao_training/**/baccZ_LL.txt')
    with open( '../GEM2_nao_training/baccZ_LL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccX_RL 7
    files = glob.glob('../GEM2_nao_training/**/baccX_RL.txt')
    with open( '../GEM2_nao_training/baccX_RL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccY_RL 8
    files = glob.glob('../GEM2_nao_training/**/baccY_RL.txt')
    with open( '../GEM2_nao_training/baccY_RL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccZ_RL 9
    files = glob.glob('../GEM2_nao_training/**/baccZ_RL.txt')
    with open( '../GEM2_nao_training/baccZ_RL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )            

    #baccX 10
    files = glob.glob('../GEM2_nao_training/**/baccX.txt')
    with open( '../GEM2_nao_training/baccX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccY 11
    files = glob.glob('../GEM2_nao_training/**/baccY.txt')
    with open( '../GEM2_nao_training/baccY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccZ 12
    files = glob.glob('../GEM2_nao_training/**/baccZ.txt')
    with open( '../GEM2_nao_training/baccZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #comvX 13
    files = glob.glob('../GEM2_nao_training/**/comvX.txt')
    with open( '../GEM2_nao_training/comvX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #comvY 14
    files = glob.glob('../GEM2_nao_training/**/comvY.txt')
    with open( '../GEM2_nao_training/comvY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #comvZ 15
    files = glob.glob('../GEM2_nao_training/**/comvZ.txt')
    with open( '../GEM2_nao_training/comvZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #lfX 16
    files = glob.glob('../GEM2_nao_training/**/lfX.txt')
    with open( '../GEM2_nao_training/lfX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #lfY 17
    files = glob.glob('../GEM2_nao_training/**/lfY.txt')
    with open( '../GEM2_nao_training/lfY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #lfZ 18
    files = glob.glob('../GEM2_nao_training/**/lfZ.txt')
    with open( '../GEM2_nao_training/lfZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rfX 19
    files = glob.glob('../GEM2_nao_training/**/rfX.txt')
    with open( '../GEM2_nao_training/rfX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rfY 20
    files = glob.glob('../GEM2_nao_training/**/rfY.txt')
    with open( '../GEM2_nao_training/rfY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #rfY 21 
    files = glob.glob('../GEM2_nao_training/**/rfZ.txt')
    with open( '../GEM2_nao_training/rfZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #ltX 22
    files = glob.glob('../GEM2_nao_training/**/ltX.txt')
    with open( '../GEM2_nao_training/ltX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #ltY 23
    files = glob.glob('../GEM2_nao_training/**/ltY.txt')
    with open( '../GEM2_nao_training/ltY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #ltZ 24
    files = glob.glob('../GEM2_nao_training/**/ltZ.txt')
    with open( '../GEM2_nao_training/ltZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rtX 25
    files = glob.glob('../GEM2_nao_training/**/rtX.txt')
    with open( '../GEM2_nao_training/rtX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rtY 26
    files = glob.glob('../GEM2_nao_training/**/rtY.txt')
    with open( '../GEM2_nao_training/rtY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #rtZ 27
    files = glob.glob('../GEM2_nao_training/**/rtZ.txt')
    with open( '../GEM2_nao_training/rtZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #laccX 28
    files = glob.glob('../GEM2_nao_training/**/laccX.txt')
    with open( '../GEM2_nao_training/laccX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #laccY 29
    files = glob.glob('../GEM2_nao_training/**/laccY.txt')
    with open( '../GEM2_nao_training/laccY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #laccZ 30 
    files = glob.glob('../GEM2_nao_training/**/laccZ.txt')
    with open( '../GEM2_nao_training/laccZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #raccX 31
    files = glob.glob('../GEM2_nao_training/**/raccX.txt')
    with open( '../GEM2_nao_training/raccX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #raccY 32
    files = glob.glob('../GEM2_nao_training/**/raccY.txt')
    with open( '../GEM2_nao_training/raccY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #raccZ 33
    files = glob.glob('../GEM2_nao_training/**/raccZ.txt')
    with open( '../GEM2_nao_training/raccZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rvX 34
    files = glob.glob('../GEM2_nao_training/**/rvX.txt')
    with open( '../GEM2_nao_training/rvX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rvY 35
    files = glob.glob('../GEM2_nao_training/**/rvY.txt')
    with open( '../GEM2_nao_training/rvY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #rvZ 35
    files = glob.glob('../GEM2_nao_training/**/rvZ.txt')
    with open( '../GEM2_nao_training/rvZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )           
    #lvX 36
    files = glob.glob('../GEM2_nao_training/**/lvX.txt')
    with open( '../GEM2_nao_training/lvX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #lvY 37
    files = glob.glob('../GEM2_nao_training/**/lvY.txt')
    with open( '../GEM2_nao_training/lvY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #lvZ 38
    files = glob.glob('../GEM2_nao_training/**/lvZ.txt')
    with open( '../GEM2_nao_training/lvZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )           


    #rwX 39
    files = glob.glob('../GEM2_nao_training/**/rwX.txt')
    with open( '../GEM2_nao_training/rwX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rwY 40
    files = glob.glob('../GEM2_nao_training/**/rwY.txt')
    with open( '../GEM2_nao_training/rwY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #rwZ 41 
    files = glob.glob('../GEM2_nao_training/**/rwZ.txt')
    with open( '../GEM2_nao_training/rwZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )           
    #lwX 42
    files = glob.glob('../GEM2_nao_training/**/lwX.txt')
    with open( '../GEM2_nao_training/lwX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #lwY 43
    files = glob.glob('../GEM2_nao_training/**/lwY.txt')
    with open( '../GEM2_nao_training/lwY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #lwZ 44
    files = glob.glob('../GEM2_nao_training/**/lwZ.txt')
    with open( '../GEM2_nao_training/lwZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )           

    #gX 45
    files = glob.glob('../GEM2_nao_training/**/gX.txt')
    with open( '../GEM2_nao_training/gX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #gY 46
    files = glob.glob('../GEM2_nao_training/**/gY.txt')
    with open( '../GEM2_nao_training/gY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #gZ 47
    files = glob.glob('../GEM2_nao_training/**/gZ.txt')
    with open( '../GEM2_nao_training/gZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #lgX 48
    files = glob.glob('../GEM2_nao_training/**/lgX.txt')
    with open( '../GEM2_nao_training/lgX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #lgY 49
    files = glob.glob('../GEM2_nao_training/**/lgY.txt')
    with open( '../GEM2_nao_training/lgY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #lgZ 50
    files = glob.glob('../GEM2_nao_training/**/lgZ.txt')
    with open( '../GEM2_nao_training/lgZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rgX 51
    files = glob.glob('../GEM2_nao_training/**/rgX.txt')
    with open( '../GEM2_nao_training/rgX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rgY 52
    files = glob.glob('../GEM2_nao_training/**/rgY.txt')
    with open( '../GEM2_nao_training/rgY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #rgZ 53
    files = glob.glob('../GEM2_nao_training/**/rgZ.txt')
    with open( '../GEM2_nao_training/rgZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )


    #baccX_LL 4
    files = glob.glob('../GEM2_nao_training/**/bgX_LL.txt')
    with open( '../GEM2_nao_training/bgX_LL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccY_LL 5 
    files = glob.glob('../GEM2_nao_training/**/bgY_LL.txt')
    with open( '../GEM2_nao_training/bgY_LL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccZ_LL 6
    files = glob.glob('../GEM2_nao_training/**/bgZ_LL.txt')
    with open( '../GEM2_nao_training/bgZ_LL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccX_RL 7
    files = glob.glob('../GEM2_nao_training/**/bgX_RL.txt')
    with open( '../GEM2_nao_training/bgX_RL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccY_RL 8
    files = glob.glob('../GEM2_nao_training/**/bgY_RL.txt')
    with open( '../GEM2_nao_training/bgY_RL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccZ_RL 9
    files = glob.glob('../GEM2_nao_training/**/bgZ_RL.txt')
    with open( '../GEM2_nao_training/bgZ_RL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )            

    #baccX 10
    files = glob.glob('../GEM2_nao_training/**/bgX.txt')
    with open( '../GEM2_nao_training/bgX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccY 11
    files = glob.glob('../GEM2_nao_training/**/bgY.txt')
    with open( '../GEM2_nao_training/bgY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccZ 12
    files = glob.glob('../GEM2_nao_training/**/bgZ.txt')
    with open( '../GEM2_nao_training/bgZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    return 0

if __name__ == "__main__":
    main()