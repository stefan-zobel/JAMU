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

/**
 * Tests for the matrix norms in all four hierarchies.
 */
public final class NormTest {

    // relative bound for the norms that accumulate; measured worst is 0.0
    private static final double TOL_D = 1.0e-15;
    private static final float TOL_F = 1.0e-6f;

    // column-major 2 x 2, every entry negative, largest magnitude 7.5
    private static final double[] ALL_NEGATIVE = { -3.0, -7.5, -0.25, -1.0 };
    // column-major 2 x 2, mixed signs, largest magnitude 9.0 is the negative one
    private static final double[] MIXED_SIGNS = { 1.0, -9.0, 2.0, 0.5 };

    private static final int[][] SHAPES = { { 1, 1 }, { 4, 4 }, { 7, 3 }, { 3, 7 }, { 20, 15 } };

    @Test
    public void testMaxAbsOfAnAllNegativeMatrix() {
        assertEquals("MatrixD", 7.5, realD(ALL_NEGATIVE).normMaxAbs(), 0.0);
        assertEquals("MatrixF", 7.5f, realF(ALL_NEGATIVE).normMaxAbs(), 0.0f);
        assertEquals("ComplexMatrixD", 7.5, complexD(ALL_NEGATIVE).normMaxAbs(), 0.0);
        assertEquals("ComplexMatrixF", 7.5f, complexF(ALL_NEGATIVE).normMaxAbs(), 0.0f);
    }

    @Test
    public void testMaxAbsWhenTheLargestMagnitudeIsNegative() {
        assertEquals("MatrixD", 9.0, realD(MIXED_SIGNS).normMaxAbs(), 0.0);
        assertEquals("MatrixF", 9.0f, realF(MIXED_SIGNS).normMaxAbs(), 0.0f);
        assertEquals("ComplexMatrixD", 9.0, complexD(MIXED_SIGNS).normMaxAbs(), 0.0);
        assertEquals("ComplexMatrixF", 9.0f, complexF(MIXED_SIGNS).normMaxAbs(), 0.0f);
    }

    @Test
    public void testMaxAbsMatchesTheLargestAbsoluteEntry() {
        for (int[] s : SHAPES) {
            String at = " at " + s[0] + "x" + s[1];
            MatrixD d = Matrices.randomUniformD(s[0], s[1], -1.0, 1.0, 17L);
            assertEquals("MatrixD" + at, maxAbs(d.getArrayUnsafe()), d.normMaxAbs(), 0.0);
            MatrixF f = Matrices.randomUniformF(s[0], s[1], -1.0f, 1.0f, 17L);
            assertEquals("MatrixF" + at, (float) maxAbs(widen(f.getArrayUnsafe())), f.normMaxAbs(), 0.0f);
            ComplexMatrixD cd = Matrices.randomUniformComplexD(s[0], s[1], -1.0, 1.0, 17L);
            assertEquals("ComplexMatrixD" + at, maxAbsComplex(cd.getArrayUnsafe()), cd.normMaxAbs(),
                    TOL_D * cd.normMaxAbs());
            ComplexMatrixF cf = Matrices.randomUniformComplexF(s[0], s[1], -1.0f, 1.0f, 17L);
            assertEquals("ComplexMatrixF" + at, (float) maxAbsComplex(widen(cf.getArrayUnsafe())),
                    cf.normMaxAbs(), TOL_F * cf.normMaxAbs());
        }
    }

    @Test
    public void testMaxAbsIsUnchangedByNegation() {
        for (int[] s : SHAPES) {
            String at = " at " + s[0] + "x" + s[1];
            MatrixD d = Matrices.randomUniformD(s[0], s[1], -1.0, 1.0, 23L);
            assertEquals("MatrixD" + at, d.normMaxAbs(), d.uminus().normMaxAbs(), 0.0);
            MatrixF f = Matrices.randomUniformF(s[0], s[1], -1.0f, 1.0f, 23L);
            assertEquals("MatrixF" + at, f.normMaxAbs(), f.uminus().normMaxAbs(), 0.0f);
            ComplexMatrixD cd = Matrices.randomUniformComplexD(s[0], s[1], -1.0, 1.0, 23L);
            assertEquals("ComplexMatrixD" + at, cd.normMaxAbs(), cd.uminus().normMaxAbs(), 0.0);
            ComplexMatrixF cf = Matrices.randomUniformComplexF(s[0], s[1], -1.0f, 1.0f, 23L);
            assertEquals("ComplexMatrixF" + at, cf.normMaxAbs(), cf.uminus().normMaxAbs(), 0.0f);
        }
    }

    @Test
    public void testTheColumnAndRowSumNormsMatchTheirDefinitions() {
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            String at = " at " + m + "x" + n;
            MatrixD d = Matrices.randomUniformD(m, n, -1.0, 1.0, 31L);
            assertEquals("MatrixD norm1" + at, colSum(d.getArrayUnsafe(), m, n, false), d.norm1(),
                    TOL_D * d.norm1());
            assertEquals("MatrixD normInf" + at, rowSum(d.getArrayUnsafe(), m, n, false), d.normInf(),
                    TOL_D * d.normInf());
            MatrixF f = Matrices.randomUniformF(m, n, -1.0f, 1.0f, 31L);
            assertEquals("MatrixF norm1" + at, (float) colSum(widen(f.getArrayUnsafe()), m, n, false),
                    f.norm1(), TOL_F * f.norm1());
            assertEquals("MatrixF normInf" + at, (float) rowSum(widen(f.getArrayUnsafe()), m, n, false),
                    f.normInf(), TOL_F * f.normInf());
            ComplexMatrixD cd = Matrices.randomUniformComplexD(m, n, -1.0, 1.0, 31L);
            assertEquals("ComplexMatrixD norm1" + at, colSum(cd.getArrayUnsafe(), m, n, true), cd.norm1(),
                    TOL_D * cd.norm1());
            assertEquals("ComplexMatrixD normInf" + at, rowSum(cd.getArrayUnsafe(), m, n, true),
                    cd.normInf(), TOL_D * cd.normInf());
            ComplexMatrixF cf = Matrices.randomUniformComplexF(m, n, -1.0f, 1.0f, 31L);
            assertEquals("ComplexMatrixF norm1" + at, colSum(widen(cf.getArrayUnsafe()), m, n, true),
                    cf.norm1(), TOL_F * cf.norm1());
            assertEquals("ComplexMatrixF normInf" + at, rowSum(widen(cf.getArrayUnsafe()), m, n, true),
                    cf.normInf(), TOL_F * cf.normInf());
        }
    }

    @Test
    public void testTheFrobeniusNormMatchesItsDefinition() {
        for (int[] s : SHAPES) {
            String at = " at " + s[0] + "x" + s[1];
            MatrixD d = Matrices.randomUniformD(s[0], s[1], -1.0, 1.0, 37L);
            assertEquals("MatrixD" + at, frobenius(d.getArrayUnsafe()), d.normF(), TOL_D * d.normF());
            MatrixF f = Matrices.randomUniformF(s[0], s[1], -1.0f, 1.0f, 37L);
            assertEquals("MatrixF" + at, (float) frobenius(widen(f.getArrayUnsafe())), f.normF(),
                    TOL_F * f.normF());
            ComplexMatrixD cd = Matrices.randomUniformComplexD(s[0], s[1], -1.0, 1.0, 37L);
            assertEquals("ComplexMatrixD" + at, frobenius(cd.getArrayUnsafe()), cd.normF(),
                    TOL_D * cd.normF());
            ComplexMatrixF cf = Matrices.randomUniformComplexF(s[0], s[1], -1.0f, 1.0f, 37L);
            assertEquals("ComplexMatrixF" + at, (float) frobenius(widen(cf.getArrayUnsafe())), cf.normF(),
                    TOL_F * cf.normF());
        }
    }

    @Test
    public void testTheOtherNormsAreUnchangedByNegation() {
        for (int[] s : SHAPES) {
            String at = " at " + s[0] + "x" + s[1];
            MatrixD d = Matrices.randomUniformD(s[0], s[1], -1.0, 1.0, 41L);
            MatrixD dn = d.uminus();
            assertEquals("MatrixD norm1" + at, d.norm1(), dn.norm1(), 0.0);
            assertEquals("MatrixD normInf" + at, d.normInf(), dn.normInf(), 0.0);
            assertEquals("MatrixD normF" + at, d.normF(), dn.normF(), 0.0);
            ComplexMatrixD cd = Matrices.randomUniformComplexD(s[0], s[1], -1.0, 1.0, 41L);
            ComplexMatrixD cdn = cd.uminus();
            assertEquals("ComplexMatrixD norm1" + at, cd.norm1(), cdn.norm1(), 0.0);
            assertEquals("ComplexMatrixD normInf" + at, cd.normInf(), cdn.normInf(), 0.0);
            assertEquals("ComplexMatrixD normF" + at, cd.normF(), cdn.normF(), 0.0);
        }
    }

    private static MatrixD realD(double[] colMajor) {
        MatrixD m = Matrices.createD(2, 2);
        for (int col = 0, k = 0; col < 2; ++col) {
            for (int row = 0; row < 2; ++row, ++k) {
                m.set(row, col, colMajor[k]);
            }
        }
        return m;
    }

    private static MatrixF realF(double[] colMajor) {
        MatrixF m = Matrices.createF(2, 2);
        for (int col = 0, k = 0; col < 2; ++col) {
            for (int row = 0; row < 2; ++row, ++k) {
                m.set(row, col, (float) colMajor[k]);
            }
        }
        return m;
    }

    private static ComplexMatrixD complexD(double[] colMajor) {
        ComplexMatrixD m = Matrices.createComplexD(2, 2);
        for (int col = 0, k = 0; col < 2; ++col) {
            for (int row = 0; row < 2; ++row, ++k) {
                m.set(row, col, colMajor[k], 0.0);
            }
        }
        return m;
    }

    private static ComplexMatrixF complexF(double[] colMajor) {
        ComplexMatrixF m = Matrices.createComplexF(2, 2);
        for (int col = 0, k = 0; col < 2; ++col) {
            for (int row = 0; row < 2; ++row, ++k) {
                m.set(row, col, (float) colMajor[k], 0.0f);
            }
        }
        return m;
    }

    private static double[] widen(float[] a) {
        double[] w = new double[a.length];
        for (int i = 0; i < a.length; ++i) {
            w[i] = a[i];
        }
        return w;
    }

    private static double maxAbs(double[] a) {
        double max = 0.0;
        for (int i = 0; i < a.length; ++i) {
            max = Math.max(max, Math.abs(a[i]));
        }
        return max;
    }

    private static double maxAbsComplex(double[] a) {
        double max = 0.0;
        for (int i = 0; i < a.length; i += 2) {
            // codeql[java/index-out-of-bounds]
            max = Math.max(max, Math.hypot(a[i], a[i + 1]));
        }
        return max;
    }

    private static double colSum(double[] a, int rows, int cols, boolean complex) {
        int w = complex ? 2 : 1;
        double max = 0.0;
        for (int col = 0; col < cols; ++col) {
            double sum = 0.0;
            for (int row = 0; row < rows; ++row) {
                sum += entry(a, w * (col * rows + row), complex);
            }
            max = Math.max(max, sum);
        }
        return max;
    }

    private static double rowSum(double[] a, int rows, int cols, boolean complex) {
        int w = complex ? 2 : 1;
        double max = 0.0;
        for (int row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (int col = 0; col < cols; ++col) {
                sum += entry(a, w * (col * rows + row), complex);
            }
            max = Math.max(max, sum);
        }
        return max;
    }

    private static double frobenius(double[] a) {
        double sum = 0.0;
        for (int i = 0; i < a.length; ++i) {
            sum += a[i] * a[i];
        }
        return Math.sqrt(sum);
    }

    private static double entry(double[] a, int idx, boolean complex) {
        return complex ? Math.hypot(a[idx], a[idx + 1]) : Math.abs(a[idx]);
    }
}
