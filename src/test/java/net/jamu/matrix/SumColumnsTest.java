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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

/**
 * Tests for {@code Matrices.sumColumns} and {@code Matrices.colsAverage} in all
 * four hierarchies. The parity tests pin the result bit for bit to the original
 * row-first accumulation, so the loop order can change but the numbers cannot.
 */
public final class SumColumnsTest {

    // column-major 2 x 3: rows (1, 3, 5) and (2, 4, 6), row sums 9 and 12
    private static final double[] TWO_BY_THREE = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

    private static final int[][] SHAPES = { { 1, 1 }, { 1, 7 }, { 7, 1 }, { 4, 4 }, { 7, 3 }, { 3, 7 },
            { 20, 15 } };

    @Test
    public void testTheRowSumsOfASmallMatrix() {
        MatrixD d = realD(2, 3, TWO_BY_THREE);
        assertArrayEquals("MatrixD", new double[] { 9.0, 12.0 }, Matrices.sumColumns(d).getArrayUnsafe(), 0.0);
        MatrixF f = realF(2, 3, TWO_BY_THREE);
        assertArrayEquals("MatrixF", new float[] { 9.0f, 12.0f }, Matrices.sumColumns(f).getArrayUnsafe(), 0.0f);
        // imaginary part is the negated real part
        ComplexMatrixD cd = complexD(2, 3, TWO_BY_THREE);
        assertArrayEquals("ComplexMatrixD", new double[] { 9.0, -9.0, 12.0, -12.0 },
                Matrices.sumColumns(cd).getArrayUnsafe(), 0.0);
        ComplexMatrixF cf = complexF(2, 3, TWO_BY_THREE);
        assertArrayEquals("ComplexMatrixF", new float[] { 9.0f, -9.0f, 12.0f, -12.0f },
                Matrices.sumColumns(cf).getArrayUnsafe(), 0.0f);
    }

    @Test
    public void testTheResultIsAColumnVectorAndTheArgumentIsUntouched() {
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            String at = " at " + m + "x" + n;
            MatrixD d = Matrices.randomUniformD(m, n, -1.0, 1.0, 11L);
            double[] dBefore = d.getArrayUnsafe().clone();
            MatrixD sd = Matrices.sumColumns(d);
            assertShape("MatrixD" + at, m, sd);
            assertArrayEquals("MatrixD" + at, dBefore, d.getArrayUnsafe(), 0.0);

            MatrixF f = Matrices.randomUniformF(m, n, -1.0f, 1.0f, 11L);
            float[] fBefore = f.getArrayUnsafe().clone();
            MatrixF sf = Matrices.sumColumns(f);
            assertShape("MatrixF" + at, m, sf);
            assertArrayEquals("MatrixF" + at, fBefore, f.getArrayUnsafe(), 0.0f);

            ComplexMatrixD cd = Matrices.randomUniformComplexD(m, n, -1.0, 1.0, 11L);
            double[] cdBefore = cd.getArrayUnsafe().clone();
            ComplexMatrixD scd = Matrices.sumColumns(cd);
            assertShape("ComplexMatrixD" + at, m, scd);
            assertArrayEquals("ComplexMatrixD" + at, cdBefore, cd.getArrayUnsafe(), 0.0);

            ComplexMatrixF cf = Matrices.randomUniformComplexF(m, n, -1.0f, 1.0f, 11L);
            float[] cfBefore = cf.getArrayUnsafe().clone();
            ComplexMatrixF scf = Matrices.sumColumns(cf);
            assertShape("ComplexMatrixF" + at, m, scf);
            assertArrayEquals("ComplexMatrixF" + at, cfBefore, cf.getArrayUnsafe(), 0.0f);
        }
    }

    @Test
    public void testMatrixDIsBitIdenticalToTheRowFirstAccumulation() {
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            MatrixD d = spread(Matrices.randomUniformD(m, n, -1.0, 1.0, 13L));
            double[] expected = rowFirstSums(d.getArrayUnsafe(), m, n, 1);
            assertBits("MatrixD at " + m + "x" + n, expected, Matrices.sumColumns(d).getArrayUnsafe());
        }
    }

    @Test
    public void testMatrixFIsBitIdenticalToTheRowFirstAccumulation() {
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            MatrixF f = spread(Matrices.randomUniformF(m, n, -1.0f, 1.0f, 13L));
            double[] expected = rowFirstSums(widen(f.getArrayUnsafe()), m, n, 1);
            assertBits("MatrixF at " + m + "x" + n, narrow(expected), Matrices.sumColumns(f).getArrayUnsafe());
        }
    }

    @Test
    public void testComplexMatrixDIsBitIdenticalToTheRowFirstAccumulation() {
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            ComplexMatrixD cd = spread(Matrices.randomUniformComplexD(m, n, -1.0, 1.0, 13L));
            double[] expected = rowFirstSums(cd.getArrayUnsafe(), m, n, 2);
            assertBits("ComplexMatrixD at " + m + "x" + n, expected, Matrices.sumColumns(cd).getArrayUnsafe());
        }
    }

    @Test
    public void testComplexMatrixFIsBitIdenticalToTheRowFirstAccumulation() {
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            ComplexMatrixF cf = spread(Matrices.randomUniformComplexF(m, n, -1.0f, 1.0f, 13L));
            double[] expected = rowFirstSums(widen(cf.getArrayUnsafe()), m, n, 2);
            assertBits("ComplexMatrixF at " + m + "x" + n, narrow(expected),
                    Matrices.sumColumns(cf).getArrayUnsafe());
        }
    }

    @Test
    public void testColsAverageIsTheRowSumDividedByTheColumnCount() {
        MatrixD d = realD(2, 3, TWO_BY_THREE);
        assertArrayEquals("MatrixD", new double[] { 3.0, 4.0 }, Matrices.colsAverage(d).getArrayUnsafe(), 0.0);
        MatrixF f = realF(2, 3, TWO_BY_THREE);
        assertArrayEquals("MatrixF", new float[] { 3.0f, 4.0f }, Matrices.colsAverage(f).getArrayUnsafe(), 0.0f);
    }

    // the reduction as it was before it became column-first
    private static double[] rowFirstSums(double[] a, int rows, int cols, int width) {
        double[] sums = new double[width * rows];
        for (int row = 0; row < rows; ++row) {
            for (int k = 0; k < width; ++k) {
                double sum = 0.0;
                for (int col = 0; col < cols; ++col) {
                    sum += a[width * (col * rows + row) + k];
                }
                sums[width * row + k] = sum;
            }
        }
        return sums;
    }

    // magnitudes from 1e-6 to 1e+6 across the columns, so that a change in the
    // order of the additions shows up in the low bits of the sums
    private static MatrixD spread(MatrixD A) {
        spread(A.getArrayUnsafe(), A.numRows(), A.numColumns(), 1);
        return A;
    }

    private static MatrixF spread(MatrixF A) {
        float[] a = A.getArrayUnsafe();
        double[] w = widen(a);
        spread(w, A.numRows(), A.numColumns(), 1);
        System.arraycopy(narrow(w), 0, a, 0, a.length);
        return A;
    }

    private static ComplexMatrixD spread(ComplexMatrixD A) {
        spread(A.getArrayUnsafe(), A.numRows(), A.numColumns(), 2);
        return A;
    }

    private static ComplexMatrixF spread(ComplexMatrixF A) {
        float[] a = A.getArrayUnsafe();
        double[] w = widen(a);
        spread(w, A.numRows(), A.numColumns(), 2);
        System.arraycopy(narrow(w), 0, a, 0, a.length);
        return A;
    }

    private static void spread(double[] a, int rows, int cols, int width) {
        for (int col = 0; col < cols; ++col) {
            double scale = Math.pow(10.0, (col % 13) - 6);
            for (int i = width * col * rows; i < width * (col + 1) * rows; ++i) {
                a[i] *= scale;
            }
        }
    }

    private static void assertShape(String msg, int rows, MatrixDimensions s) {
        assertEquals(msg + " rows", rows, s.numRows());
        assertEquals(msg + " cols", 1, s.numColumns());
    }

    private static void assertBits(String msg, double[] expected, double[] actual) {
        assertEquals(msg + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            assertEquals(msg + " [" + i + "]", Double.doubleToRawLongBits(expected[i]),
                    Double.doubleToRawLongBits(actual[i]));
        }
    }

    private static void assertBits(String msg, float[] expected, float[] actual) {
        assertEquals(msg + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            assertEquals(msg + " [" + i + "]", Float.floatToRawIntBits(expected[i]),
                    Float.floatToRawIntBits(actual[i]));
        }
    }

    private static MatrixD realD(int rows, int cols, double[] values) {
        MatrixD m = Matrices.createD(rows, cols);
        System.arraycopy(values, 0, m.getArrayUnsafe(), 0, values.length);
        return m;
    }

    private static MatrixF realF(int rows, int cols, double[] values) {
        MatrixF m = Matrices.createF(rows, cols);
        float[] a = m.getArrayUnsafe();
        for (int i = 0; i < values.length; ++i) {
            a[i] = (float) values[i];
        }
        return m;
    }

    private static ComplexMatrixD complexD(int rows, int cols, double[] values) {
        ComplexMatrixD m = Matrices.createComplexD(rows, cols);
        double[] a = m.getArrayUnsafe();
        for (int i = 0; i < values.length; ++i) {
            a[2 * i] = values[i];
            a[2 * i + 1] = -values[i];
        }
        return m;
    }

    private static ComplexMatrixF complexF(int rows, int cols, double[] values) {
        ComplexMatrixF m = Matrices.createComplexF(rows, cols);
        float[] a = m.getArrayUnsafe();
        for (int i = 0; i < values.length; ++i) {
            a[2 * i] = (float) values[i];
            a[2 * i + 1] = (float) -values[i];
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

    private static float[] narrow(double[] a) {
        float[] n = new float[a.length];
        for (int i = 0; i < a.length; ++i) {
            n[i] = (float) a[i];
        }
        return n;
    }
}
