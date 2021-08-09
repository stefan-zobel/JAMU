/*
 * Copyright 2021 Stefan Zobel
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.jamu.matrix;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Dynamic programming solution of the matrix-chain multiplication problem as
 * outlined in Cormen et al. (chapter 15.2).
 * <p>
 * See https://en.wikipedia.org/wiki/Matrix_chain_multiplication and Abdul
 * Bari's YouTube video https://www.youtube.com/watch?v=_WncuhSJZyA
 */
final class MatrixChain {

    private final long[] d;

    MatrixChain(Dimensions A0, Dimensions A1, Dimensions[] Ai) {
        long[] dims = new long[Ai.length + 3];
        dims[0] = A0.numRows();
        dims[1] = A0.numColumns();
        dims[2] = A1.numColumns();
        for (int i = 3; i < dims.length; ++i) {
            dims[i] = Ai[i - 3].numColumns(); // "lgtm[java/index-out-of-bounds]"
        }
        d = dims;
    }

    UPInts computeSplits() {
        int n = d.length - 1;
        UPInts splits = new UPInts(n);
        UPLongs costs = new UPLongs(n);
        for (int l = 1; l < n; ++l) {
            for (int i = 0; i < n - l; ++i) {
                int j = i + l;
                costs.set(i, j, Long.MAX_VALUE);
                for (int k = i; k < j; ++k) {
                    long cost = costs.get(i, k) + costs.get(k + 1, j) + d[i] * d[k + 1] * d[j + 1];
                    if (cost < costs.get(i, j)) {
                        costs.set(i, j, cost);
                        splits.set(i, j, k);
                    }
                }
            }
        }
        return splits;
    }

    static ArrayList<MatrixD> buildList(MatrixD A0, MatrixD A1, MatrixD[] Ai) {
        ArrayList<MatrixD> mList = new ArrayList<>(Ai.length + 2);
        mList.add(A0);
        mList.add(A1);
        mList.addAll(Arrays.asList(Ai));
        return mList;
    }

    static ArrayList<MatrixF> buildList(MatrixF A0, MatrixF A1, MatrixF[] Ai) {
        ArrayList<MatrixF> mList = new ArrayList<>(Ai.length + 2);
        mList.add(A0);
        mList.add(A1);
        mList.addAll(Arrays.asList(Ai));
        return mList;
    }

    static ArrayList<ComplexMatrixD> buildList(ComplexMatrixD A0, ComplexMatrixD A1, ComplexMatrixD[] Ai) {
        ArrayList<ComplexMatrixD> mList = new ArrayList<>(Ai.length + 2);
        mList.add(A0);
        mList.add(A1);
        mList.addAll(Arrays.asList(Ai));
        return mList;
    }

    static ArrayList<ComplexMatrixF> buildList(ComplexMatrixF A0, ComplexMatrixF A1, ComplexMatrixF[] Ai) {
        ArrayList<ComplexMatrixF> mList = new ArrayList<>(Ai.length + 2);
        mList.add(A0);
        mList.add(A1);
        mList.addAll(Arrays.asList(Ai));
        return mList;
    }
}
