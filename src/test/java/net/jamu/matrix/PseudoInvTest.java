/*
 * Copyright 2026 Stefan Zobel
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

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

/**
 * Tests for pseudoInv and inverse.
 */
public final class PseudoInvTest {

    private static final double TOL = 1.0e-13;

    /** an m-by-n matrix with exactly the given rank */
    private static MatrixD withRank(int m, int n, int rank, long seed) {
        MatrixD V = Matrices.randomNormalD(m, m, seed).qrd().getQ();
        MatrixD W = Matrices.randomNormalD(n, n, seed + 1L).qrd().getQ();
        double[] s = new double[Math.min(m, n)];
        for (int i = 0; i < rank; ++i) {
            s[i] = 10.0 / (i + 1);
        }
        return V.times(Matrices.diagD(m, n, s)).times(W.transpose());
    }

    /** checks all four Moore-Penrose conditions */
    private static void assertPseudoInverse(String what, MatrixD A) {
        MatrixD P = A.pseudoInv();
        MatrixD AP = A.times(P);
        MatrixD PA = P.times(A);
        assertTrue(what + ": A P A = A",
                A.copy().addInplace(-1.0, A.times(P).times(A)).normF() / A.normF() <= TOL);
        assertTrue(what + ": P A P = P",
                P.copy().addInplace(-1.0, P.times(A).times(P)).normF() / P.normF() <= TOL);
        assertTrue(what + ": A P hermitian",
                AP.copy().addInplace(-1.0, AP.transpose()).normF() / AP.normF() <= TOL);
        assertTrue(what + ": P A hermitian",
                PA.copy().addInplace(-1.0, PA.transpose()).normF() / PA.normF() <= TOL);
    }

    @Test
    public void testSquareOfEveryRank() {
        for (int rank : new int[] { 1, 5, 30, 60 }) {
            assertPseudoInverse("60x60 rank " + rank, withRank(60, 60, rank, 21L));
        }
    }

    @Test
    public void testSquareExactlySingular() {
        // column 5 repeats column 0, so the LU has an exact zero pivot
        MatrixD A = Matrices.createD(6, 6);
        for (int j = 0; j < 6; ++j) {
            for (int i = 0; i < 6; ++i) {
                A.set(i, j, (j == 5) ? (i + 1) : ((j + 1) * (i + 1)));
            }
        }
        assertPseudoInverse("exactly singular", A);
    }

    @Test
    public void testNonSquare() {
        for (int rank : new int[] { 1, 5, 40 }) {
            assertPseudoInverse("60x40 rank " + rank, withRank(60, 40, rank, 31L));
            assertPseudoInverse("40x60 rank " + rank, withRank(40, 60, rank, 41L));
        }
    }

    @Test
    public void testScaleInvariance() {
        for (double scale : new double[] { 1.0e+100, 1.0, 1.0e-20, 1.0e-150 }) {
            assertPseudoInverse("scale " + scale, withRank(60, 60, 5, 21L).scaleInplace(scale));
        }
    }

    @Test
    public void testInverseStillRejectsNonSquare() {
        try {
            Matrices.randomNormalD(5, 4, 1L).inverse();
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // as documented
        }
    }
}
