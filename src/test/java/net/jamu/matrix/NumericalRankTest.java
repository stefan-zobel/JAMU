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

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import net.jamu.complex.Zd;
import net.jamu.complex.ZdImpl;
import net.jamu.complex.Zf;
import net.jamu.complex.ZfImpl;

/**
 * Tests for Matrices.numericalRank, one matrix per case rescaled over the
 * exponent range the SVD still resolves.
 */
public final class NumericalRankTest {

    private static final int M = 60;
    private static final int N = 60;
    private static final int RANK = 5;

    /** the scales a double SVD still resolves, subnormals excluded */
    private static final double[] SCALES_D = { 1.0e+300, 1.0e+100, 1.0, 1.0e-17, 1.0e-100, 1.0e-300 };
    /** the same for single precision */
    private static final float[] SCALES_F = { 1.0e+30f, 1.0f, 1.0e-9f, 1.0e-20f, 1.0e-30f };

    private static double[] spectrum(int len, int rank) {
        double[] s = new double[len];
        for (int i = 0; i < rank; ++i) {
            s[i] = 10.0 / (i + 1);
        }
        return s;
    }

    private static MatrixD withRankD(int m, int n, int rank, long seed) {
        MatrixD V = Matrices.randomNormalD(m, m, seed).qrd().getQ();
        MatrixD W = Matrices.randomNormalD(n, n, seed + 1L).qrd().getQ();
        return V.times(Matrices.diagD(m, n, spectrum(Math.min(m, n), rank))).times(W.transpose());
    }

    private static MatrixF withRankF(int m, int n, int rank, long seed) {
        MatrixF V = Matrices.randomNormalF(m, m, seed).qrd().getQ();
        MatrixF W = Matrices.randomNormalF(n, n, seed + 1L).qrd().getQ();
        double[] s = spectrum(Math.min(m, n), rank);
        float[] d = new float[s.length];
        for (int i = 0; i < s.length; ++i) {
            d[i] = (float) s[i];
        }
        return V.times(Matrices.diagF(m, n, d)).times(W.transpose());
    }

    private static ComplexMatrixD withRankComplexD(int m, int n, int rank, long seed) {
        ComplexMatrixD V = Matrices.randomNormalComplexD(m, m, seed).qrd().getQ();
        ComplexMatrixD W = Matrices.randomNormalComplexD(n, n, seed + 1L).qrd().getQ();
        double[] s = spectrum(Math.min(m, n), rank);
        Zd[] d = new Zd[s.length];
        for (int i = 0; i < s.length; ++i) {
            d[i] = new ZdImpl(s[i], 0.0);
        }
        ComplexMatrixD VD = V.times(Matrices.diagComplexD(m, n, d));
        return VD.conjTransBmult(W, Matrices.createComplexD(m, n));
    }

    private static ComplexMatrixF withRankComplexF(int m, int n, int rank, long seed) {
        ComplexMatrixF V = Matrices.randomNormalComplexF(m, m, seed).qrd().getQ();
        ComplexMatrixF W = Matrices.randomNormalComplexF(n, n, seed + 1L).qrd().getQ();
        double[] s = spectrum(Math.min(m, n), rank);
        Zf[] d = new Zf[s.length];
        for (int i = 0; i < s.length; ++i) {
            d[i] = new ZfImpl((float) s[i], 0.0f);
        }
        ComplexMatrixF VD = V.times(Matrices.diagComplexF(m, n, d));
        return VD.conjTransBmult(W, Matrices.createComplexF(m, n));
    }

    @Test
    public void testScaleInvarianceD() {
        for (double scale : SCALES_D) {
            assertEquals("scale " + scale, RANK,
                    Matrices.numericalRank(withRankD(M, N, RANK, 21L).scaleInplace(scale)));
        }
    }

    @Test
    public void testScaleInvarianceF() {
        for (float scale : SCALES_F) {
            assertEquals("scale " + scale, RANK,
                    Matrices.numericalRank(withRankF(M, N, RANK, 21L).scaleInplace(scale)));
        }
    }

    @Test
    public void testScaleInvarianceComplexD() {
        for (double scale : SCALES_D) {
            assertEquals("scale " + scale, RANK,
                    Matrices.numericalRank(withRankComplexD(M, N, RANK, 21L).scaleInplace(scale, 0.0)));
        }
    }

    @Test
    public void testScaleInvarianceComplexF() {
        for (float scale : SCALES_F) {
            assertEquals("scale " + scale, RANK,
                    Matrices.numericalRank(withRankComplexF(M, N, RANK, 21L).scaleInplace(scale, 0.0f)));
        }
    }

    @Test
    public void testZeroMatrixHasRankZero() {
        assertEquals(0, Matrices.numericalRank(Matrices.createD(8, 8)));
        assertEquals(0, Matrices.numericalRank(Matrices.createF(8, 8)));
        assertEquals(0, Matrices.numericalRank(Matrices.createComplexD(8, 8)));
        assertEquals(0, Matrices.numericalRank(Matrices.createComplexF(8, 8)));
    }

    @Test
    public void testExactRankAtUnitScale() {
        for (int rank : new int[] { 1, 5, 60 }) {
            assertEquals("60x60 rank " + rank, rank, Matrices.numericalRank(withRankD(60, 60, rank, 7L)));
        }
        for (int rank : new int[] { 1, 5, 40 }) {
            assertEquals("60x40 rank " + rank, rank, Matrices.numericalRank(withRankD(60, 40, rank, 7L)));
            assertEquals("40x60 rank " + rank, rank, Matrices.numericalRank(withRankD(40, 60, rank, 9L)));
        }
    }
}
