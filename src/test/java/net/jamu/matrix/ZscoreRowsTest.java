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
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;

import java.util.Random;

import org.junit.Test;

/**
 * Tests for {@code Statistics.zscoreRows} and
 * {@code Statistics.zscoreRowsInplace} on {@code MatrixD} and {@code MatrixF}.
 * The parity tests pin the z-scored matrix and both reported moments bit for
 * bit (NaNs compared as NaN) to the original row-at-a-time implementation,
 * across regimes that reach every branch of the overflow-resistant
 * accumulation, including NaN and infinite entries. The loop order can change,
 * the numbers cannot.
 */
public final class ZscoreRowsTest {

    private static final int[][] SHAPES = { { 1, 2 }, { 1, 7 }, { 2, 2 }, { 7, 3 }, { 3, 7 }, { 20, 15 },
            { 200, 50 } };

    // number of data regimes, see valueD / valueF
    private static final int REGIMES = 10;

    @Test
    public void testMatrixDIsBitIdenticalToTheRowAtATimeOriginal() {
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            for (int shift = 0; shift < REGIMES; ++shift) {
                String at = " at " + m + "x" + n + " shift " + shift;
                double[] data = dataD(m, n, shift, 17L + shift);

                double[] expected = data.clone();
                double[][] moments = rowAtATimeZscore(expected, m, n);

                MatrixD A = matrixD(m, n, data);
                Statistics.MomentsD md = new Statistics.MomentsD();
                assertSame("MatrixD result" + at, A, Statistics.zscoreRowsInplace(A, md));
                assertBits("MatrixD" + at, expected, A.getArrayUnsafe());
                assertBits("MatrixD means" + at, moments[0], md.means.getArrayUnsafe());
                assertBits("MatrixD variances" + at, moments[1], md.variances.getArrayUnsafe());

                MatrixD B = matrixD(m, n, data);
                assertSame("MatrixD result without moments" + at, B, Statistics.zscoreRowsInplace(B));
                assertBits("MatrixD without moments" + at, expected, B.getArrayUnsafe());
            }
        }
    }

    @Test
    public void testMatrixFIsBitIdenticalToTheRowAtATimeOriginal() {
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            for (int shift = 0; shift < REGIMES; ++shift) {
                String at = " at " + m + "x" + n + " shift " + shift;
                float[] data = dataF(m, n, shift, 17L + shift);

                float[] expected = data.clone();
                float[][] moments = rowAtATimeZscore(expected, m, n);

                MatrixF A = matrixF(m, n, data);
                Statistics.MomentsF mf = new Statistics.MomentsF();
                assertSame("MatrixF result" + at, A, Statistics.zscoreRowsInplace(A, mf));
                assertBits("MatrixF" + at, expected, A.getArrayUnsafe());
                assertBits("MatrixF means" + at, moments[0], mf.means.getArrayUnsafe());
                assertBits("MatrixF variances" + at, moments[1], mf.variances.getArrayUnsafe());

                MatrixF B = matrixF(m, n, data);
                assertSame("MatrixF result without moments" + at, B, Statistics.zscoreRowsInplace(B));
                assertBits("MatrixF without moments" + at, expected, B.getArrayUnsafe());
            }
        }
    }

    @Test
    public void testTheCopyingVariantLeavesTheArgumentUntouched() {
        double[] data = dataD(20, 15, 0, 3L);
        MatrixD A = matrixD(20, 15, data);
        MatrixD Z = Statistics.zscoreRows(A);
        assertNotSame("MatrixD", A, Z);
        assertBits("MatrixD argument", data, A.getArrayUnsafe());
        double[] expected = data.clone();
        rowAtATimeZscore(expected, 20, 15);
        assertBits("MatrixD result", expected, Z.getArrayUnsafe());

        float[] fdata = dataF(20, 15, 0, 3L);
        MatrixF F = matrixF(20, 15, fdata);
        MatrixF FZ = Statistics.zscoreRows(F);
        assertNotSame("MatrixF", F, FZ);
        assertBits("MatrixF argument", fdata, F.getArrayUnsafe());
        float[] fexpected = fdata.clone();
        rowAtATimeZscore(fexpected, 20, 15);
        assertBits("MatrixF result", fexpected, FZ.getArrayUnsafe());
    }

    @Test
    public void testCorrectlyShapedMomentsAreFilledInPlace() {
        MatrixD means = Matrices.createD(7, 1);
        MatrixD variances = Matrices.createD(7, 1);
        Statistics.MomentsD md = new Statistics.MomentsD(means, variances);
        double[] data = dataD(7, 3, 0, 5L);
        Statistics.zscoreRowsInplace(matrixD(7, 3, data), md);
        assertSame("MatrixD means", means, md.means);
        assertSame("MatrixD variances", variances, md.variances);
        double[][] expected = rowAtATimeZscore(data.clone(), 7, 3);
        assertBits("MatrixD means", expected[0], means.getArrayUnsafe());
        assertBits("MatrixD variances", expected[1], variances.getArrayUnsafe());

        MatrixF fmeans = Matrices.createF(7, 1);
        MatrixF fvariances = Matrices.createF(7, 1);
        Statistics.MomentsF mf = new Statistics.MomentsF(fmeans, fvariances);
        float[] fdata = dataF(7, 3, 0, 5L);
        Statistics.zscoreRowsInplace(matrixF(7, 3, fdata), mf);
        assertSame("MatrixF means", fmeans, mf.means);
        assertSame("MatrixF variances", fvariances, mf.variances);
        float[][] fexpected = rowAtATimeZscore(fdata.clone(), 7, 3);
        assertBits("MatrixF means", fexpected[0], fmeans.getArrayUnsafe());
        assertBits("MatrixF variances", fexpected[1], fvariances.getArrayUnsafe());
    }

    @Test
    public void testWronglyShapedMomentsAreReplaced() {
        MatrixD means = Matrices.createD(8, 1);
        MatrixD variances = Matrices.createD(1, 7);
        Statistics.MomentsD md = new Statistics.MomentsD(means, variances);
        double[] data = dataD(7, 3, 0, 5L);
        Statistics.zscoreRowsInplace(matrixD(7, 3, data), md);
        assertNotSame("MatrixD means", means, md.means);
        assertNotSame("MatrixD variances", variances, md.variances);
        assertShape("MatrixD means", 7, md.means);
        assertShape("MatrixD variances", 7, md.variances);
        double[][] expected = rowAtATimeZscore(data.clone(), 7, 3);
        assertBits("MatrixD means", expected[0], md.means.getArrayUnsafe());
        assertBits("MatrixD variances", expected[1], md.variances.getArrayUnsafe());

        MatrixF fmeans = Matrices.createF(8, 1);
        MatrixF fvariances = Matrices.createF(1, 7);
        Statistics.MomentsF mf = new Statistics.MomentsF(fmeans, fvariances);
        float[] fdata = dataF(7, 3, 0, 5L);
        Statistics.zscoreRowsInplace(matrixF(7, 3, fdata), mf);
        assertNotSame("MatrixF means", fmeans, mf.means);
        assertNotSame("MatrixF variances", fvariances, mf.variances);
        assertShape("MatrixF means", 7, mf.means);
        assertShape("MatrixF variances", 7, mf.variances);
        float[][] fexpected = rowAtATimeZscore(fdata.clone(), 7, 3);
        assertBits("MatrixF means", fexpected[0], mf.means.getArrayUnsafe());
        assertBits("MatrixF variances", fexpected[1], mf.variances.getArrayUnsafe());
    }

    @Test
    public void testAliasedMomentsEndUpHoldingTheVariances() {
        // today each row writes its mean first and its variance second
        MatrixD both = Matrices.createD(7, 1);
        double[] data = dataD(7, 3, 0, 5L);
        Statistics.zscoreRowsInplace(matrixD(7, 3, data), new Statistics.MomentsD(both, both));
        assertBits("MatrixD", rowAtATimeZscore(data.clone(), 7, 3)[1], both.getArrayUnsafe());

        MatrixF fboth = Matrices.createF(7, 1);
        float[] fdata = dataF(7, 3, 0, 5L);
        Statistics.zscoreRowsInplace(matrixF(7, 3, fdata), new Statistics.MomentsF(fboth, fboth));
        assertBits("MatrixF", rowAtATimeZscore(fdata.clone(), 7, 3)[1], fboth.getArrayUnsafe());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testAMatrixDColumnVectorIsRejected() {
        Statistics.zscoreRowsInplace(Matrices.createD(5, 1), new Statistics.MomentsD());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testAMatrixFColumnVectorIsRejected() {
        Statistics.zscoreRowsInplace(Matrices.createF(5, 1), new Statistics.MomentsF());
    }

    // the algorithm as it was before it became column-first, row at a time;
    // returns { means, variances }
    private static double[][] rowAtATimeZscore(double[] _a, int rows_, int cols_) {
        double[] means = new double[rows_];
        double[] variances = new double[rows_];
        for (int row = 0; row < rows_; ++row) {
            double k = _a[row];
            int count = 0;
            double shiftMean = 0.0;
            double scale = 0.0;
            double sumsquared = 1.0;
            for (int i = row; i < row + rows_ * cols_; i += rows_) {
                ++count;
                double xi = _a[i] - k;
                shiftMean = (((count - 1) * shiftMean) + xi) / count;
                if (xi != 0.0) {
                    double absxi = Math.abs(xi);
                    if (scale < absxi) {
                        double unsquared = scale / absxi;
                        sumsquared = 1.0 + sumsquared * (unsquared * unsquared);
                        scale = absxi;
                    } else {
                        double unsquared = absxi / scale;
                        sumsquared = sumsquared + (unsquared * unsquared);
                    }
                }
            }
            double mean = k + shiftMean;
            double y = (scale != 0.0) ? shiftMean / scale : shiftMean;
            double sd = scale * Math.sqrt(sumsquared / cols_ - y * y);
            double stddev = (sd == 0.0 || Double.isNaN(sd)) ? 1.0 : sd;
            for (int i = row; i < row + rows_ * cols_; i += rows_) {
                double xi = _a[i];
                xi = (xi - mean) / stddev;
                _a[i] = xi;
            }
            means[row] = mean;
            variances[row] = sd * sd;
        }
        return new double[][] { means, variances };
    }

    private static float[][] rowAtATimeZscore(float[] _a, int rows_, int cols_) {
        float[] means = new float[rows_];
        float[] variances = new float[rows_];
        for (int row = 0; row < rows_; ++row) {
            float k = _a[row];
            int count = 0;
            float shiftMean = 0.0f;
            float scale = 0.0f;
            float sumsquared = 1.0f;
            for (int i = row; i < row + rows_ * cols_; i += rows_) {
                ++count;
                float xi = _a[i] - k;
                shiftMean = (((count - 1) * shiftMean) + xi) / count;
                if (xi != 0.0f) {
                    float absxi = Math.abs(xi);
                    if (scale < absxi) {
                        float unsquared = scale / absxi;
                        sumsquared = 1.0f + sumsquared * (unsquared * unsquared);
                        scale = absxi;
                    } else {
                        float unsquared = absxi / scale;
                        sumsquared = sumsquared + (unsquared * unsquared);
                    }
                }
            }
            float mean = k + shiftMean;
            float y = (scale != 0.0f) ? shiftMean / scale : shiftMean;
            float sd = scale * (float) Math.sqrt(sumsquared / cols_ - y * y);
            float stddev = (sd == 0.0f || Float.isNaN(sd)) ? 1.0f : sd;
            for (int i = row; i < row + rows_ * cols_; i += rows_) {
                float xi = _a[i];
                xi = (xi - mean) / stddev;
                _a[i] = xi;
            }
            means[row] = mean;
            variances[row] = sd * sd;
        }
        return new float[][] { means, variances };
    }

    // column-major data where row r follows regime (r + shift) % REGIMES, so
    // that every shape with several rows interleaves different regimes
    private static double[] dataD(int rows, int cols, int shift, long seed) {
        Random rnd = new Random(seed);
        double[] a = new double[rows * cols];
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                double u = 2.0 * rnd.nextDouble() - 1.0;
                a[col * rows + row] = valueD((row + shift) % REGIMES, col, cols, u);
            }
        }
        return a;
    }

    private static double valueD(int regime, int col, int cols, double u) {
        switch (regime) {
        case 0: // plain
            return u;
        case 1: // magnitudes spread across the columns
            return u * Math.pow(10.0, (col % 13) - 6);
        case 2: // large offset, small spread
            return 1.0e8 + u;
        case 3: // large magnitudes, differences still finite
            return u * 1.0e300;
        case 4: // differences overflow to infinity
            return (col % 2 == 0 ? 1.7e308 : -1.7e308) * Math.abs(u);
        case 5: // constant row
            return 7.5;
        case 6: // NaN as the first entry, the shift reference
            return col == 0 ? Double.NaN : u;
        case 7: // NaN inside the row
            return col == cols / 2 ? Double.NaN : u;
        case 8: // infinity as the last entry
            return col == cols - 1 ? Double.POSITIVE_INFINITY : u;
        default: // signed zeros
            return col % 2 == 0 ? 0.0 : -0.0;
        }
    }

    private static float[] dataF(int rows, int cols, int shift, long seed) {
        Random rnd = new Random(seed);
        float[] a = new float[rows * cols];
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                float u = 2.0f * rnd.nextFloat() - 1.0f;
                a[col * rows + row] = valueF((row + shift) % REGIMES, col, cols, u);
            }
        }
        return a;
    }

    private static float valueF(int regime, int col, int cols, float u) {
        switch (regime) {
        case 0:
            return u;
        case 1:
            return u * (float) Math.pow(10.0, (col % 7) - 3);
        case 2:
            return 1.0e3f + u;
        case 3:
            return u * 1.0e30f;
        case 4:
            return (col % 2 == 0 ? 3.3e38f : -3.3e38f) * Math.abs(u);
        case 5:
            return 7.5f;
        case 6:
            return col == 0 ? Float.NaN : u;
        case 7:
            return col == cols / 2 ? Float.NaN : u;
        case 8:
            return col == cols - 1 ? Float.POSITIVE_INFINITY : u;
        default:
            return col % 2 == 0 ? 0.0f : -0.0f;
        }
    }

    private static MatrixD matrixD(int rows, int cols, double[] data) {
        MatrixD m = Matrices.createD(rows, cols);
        System.arraycopy(data, 0, m.getArrayUnsafe(), 0, data.length);
        return m;
    }

    private static MatrixF matrixF(int rows, int cols, float[] data) {
        MatrixF m = Matrices.createF(rows, cols);
        System.arraycopy(data, 0, m.getArrayUnsafe(), 0, data.length);
        return m;
    }

    private static void assertShape(String msg, int rows, MatrixDimensions s) {
        assertEquals(msg + " rows", rows, s.numRows());
        assertEquals(msg + " cols", 1, s.numColumns());
    }

    // Bit for bit, except that all NaNs count as one: the sign and payload of a
    // NaN depend on whether Math.abs ran interpreted or as a JIT intrinsic, so
    // raw NaN bits would make the test flaky without saying anything about the
    // algorithm.
    private static void assertBits(String msg, double[] expected, double[] actual) {
        assertEquals(msg + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            assertEquals(msg + " [" + i + "]", Double.doubleToLongBits(expected[i]),
                    Double.doubleToLongBits(actual[i]));
        }
    }

    private static void assertBits(String msg, float[] expected, float[] actual) {
        assertEquals(msg + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            assertEquals(msg + " [" + i + "]", Float.floatToIntBits(expected[i]), Float.floatToIntBits(actual[i]));
        }
    }
}
