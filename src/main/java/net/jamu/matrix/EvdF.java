/*
 * Copyright 2020 Stefan Zobel
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

import net.dedekind.lapack.Lapack;
import net.frobenius.TEigJob;
import net.frobenius.lapack.PlainLapack;
import net.jamu.complex.ZArrayUtil;
import net.jamu.complex.Zf;

/**
 * Eigenvalues and eigenvectors of a general n-by-n float matrix
 * ({@link MatrixF}).
 */
public class EvdF {

    // never compute left eigenvectors
    private static final TEigJob leftEVec = TEigJob.VALUES_ONLY;
    // caller decides whether right eigenvectors should be computed
    private final TEigJob rightEVec;
    // (right) eigenvectors if full == true or null
    private final SimpleMatrixF eigenVectors;
    // eigenvalues
    private Zf[] complexEigenValues = new Zf[] {};

    /**
     * Returns the eigenvalues.
     * 
     * @return array containing the eigenvalues
     */
    public Zf[] getEigenvalues() {
        return complexEigenValues;
    }

    /**
     * The (right) eigenvectors or {@code null} if the eigenvectors haven't been
     * computed.
     * 
     * @return n-by-n eigenvector matrix or {@code null} if
     *         {@link #hasEigenvectors()} returns {@code false}
     */
    public MatrixF getEigenvectors() {
        return eigenVectors;
    }

    /**
     * {@code true} if eigenvectors have been computed, {@code false} otherwise.
     * 
     * @return whether eigenvectors have been computed or not
     */
    public boolean hasEigenvectors() {
        return rightEVec == TEigJob.ALL;
    }

    /* package */ EvdF(MatrixF A, boolean full) {
        if (!A.isSquareMatrix()) {
            throw new IllegalArgumentException("EVD only works for square matrices");
        }
        int n = A.numRows();
        rightEVec = full ? TEigJob.ALL : TEigJob.VALUES_ONLY;
        eigenVectors = full ? new SimpleMatrixF(n, n) : null;
        computeEvdInplace(A);
    }

    private void computeEvdInplace(MatrixF A) {
        MatrixF AA = A.copy();
        int n = AA.numRows();
        int ld = Math.max(1, n);
        float[] eigValRealParts = new float[n];
        float[] eigValImagParts = new float[n];
        PlainLapack.sgeev(Lapack.getInstance(), leftEVec, rightEVec, n, AA.getArrayUnsafe(), ld, eigValRealParts,
                eigValImagParts, new float[0], ld, hasEigenvectors() ? eigenVectors.getArrayUnsafe() : new float[0],
                ld);
        // convert LAPACK eigenvalues to an Zf[] array
        complexEigenValues = ZArrayUtil.primitiveToComplexArray(eigValRealParts, eigValImagParts);
    }
}
