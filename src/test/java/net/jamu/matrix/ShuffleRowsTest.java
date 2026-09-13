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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

/**
 * Tests for {@code Statistics.shuffleRows} and
 * {@code Statistics.shuffleRowsInplace}. The seeded tests pin the result bit for
 * bit to the original implementation, which swapped whole rows in place, so the
 * way the permutation gets applied can change but not the permutation a seed
 * produces.
 */
public final class ShuffleRowsTest {

    private static final int[][] SHAPES = { { 1, 1 }, { 1, 7 }, { 7, 1 }, { 2, 2 }, { 7, 3 }, { 3, 7 },
            { 20, 15 }, { 200, 50 } };

    private static final long[] SEEDS = { 0L, 1L, 42L, -7L, 123456789L };

    @Test
    public void testMatrixDSeededShuffleMatchesTheRowSwappingOriginal() {
        for (int[] s : SHAPES) {
            for (long seed : SEEDS) {
                String at = " at " + s[0] + "x" + s[1] + " seed " + seed;
                MatrixD A = Matrices.randomUniformD(s[0], s[1], -1.0, 1.0, 5L);
                double[] expected = A.getArrayUnsafe().clone();
                rowSwappingShuffle(expected, s[0], s[1], new XoShiRo256StarStar(seed));
                MatrixD B = Statistics.shuffleRowsInplace(A, seed);
                assertBits("MatrixD" + at, expected, B.getArrayUnsafe());
            }
        }
    }

    @Test
    public void testMatrixFSeededShuffleMatchesTheRowSwappingOriginal() {
        for (int[] s : SHAPES) {
            for (long seed : SEEDS) {
                String at = " at " + s[0] + "x" + s[1] + " seed " + seed;
                MatrixF A = Matrices.randomUniformF(s[0], s[1], -1.0f, 1.0f, 5L);
                float[] expected = A.getArrayUnsafe().clone();
                rowSwappingShuffle(expected, s[0], s[1], new XoShiRo256StarStar(seed));
                MatrixF B = Statistics.shuffleRowsInplace(A, seed);
                assertBits("MatrixF" + at, expected, B.getArrayUnsafe());
            }
        }
    }

    @Test
    public void testTheSeededShuffleActuallyMovesRows() {
        MatrixD A = Matrices.randomUniformD(200, 50, -1.0, 1.0, 5L);
        double[] before = A.getArrayUnsafe().clone();
        assertFalse("MatrixD", Arrays.equals(before, Statistics.shuffleRowsInplace(A, 42L).getArrayUnsafe()));
        MatrixF F = Matrices.randomUniformF(200, 50, -1.0f, 1.0f, 5L);
        float[] fBefore = F.getArrayUnsafe().clone();
        assertFalse("MatrixF", Arrays.equals(fBefore, Statistics.shuffleRowsInplace(F, 42L).getArrayUnsafe()));
    }

    @Test
    public void testEveryVariantIsAPermutationOfTheRows() {
        for (int[] s : SHAPES) {
            String at = " at " + s[0] + "x" + s[1];
            MatrixD A = Matrices.randomUniformD(s[0], s[1], -1.0, 1.0, 9L);
            assertTrue("shuffleRows(MatrixD)" + at, sameRows(A, Statistics.shuffleRows(A)));
            assertTrue("shuffleRowsInplace(MatrixD)" + at, sameRows(A, Statistics.shuffleRowsInplace(A.copy())));
            assertTrue("shuffleRowsInplace(MatrixD, seed)" + at,
                    sameRows(A, Statistics.shuffleRowsInplace(A.copy(), 3L)));
            MatrixF F = Matrices.randomUniformF(s[0], s[1], -1.0f, 1.0f, 9L);
            assertTrue("shuffleRows(MatrixF)" + at, sameRows(F, Statistics.shuffleRows(F)));
            assertTrue("shuffleRowsInplace(MatrixF)" + at, sameRows(F, Statistics.shuffleRowsInplace(F.copy())));
            assertTrue("shuffleRowsInplace(MatrixF, seed)" + at,
                    sameRows(F, Statistics.shuffleRowsInplace(F.copy(), 3L)));
        }
    }

    @Test
    public void testTheCopyingVariantLeavesTheArgumentUntouched() {
        MatrixD A = Matrices.randomUniformD(20, 15, -1.0, 1.0, 11L);
        double[] before = A.getArrayUnsafe().clone();
        Statistics.shuffleRows(A);
        assertArrayEquals("MatrixD", before, A.getArrayUnsafe(), 0.0);
        MatrixF F = Matrices.randomUniformF(20, 15, -1.0f, 1.0f, 11L);
        float[] fBefore = F.getArrayUnsafe().clone();
        Statistics.shuffleRows(F);
        assertArrayEquals("MatrixF", fBefore, F.getArrayUnsafe(), 0.0f);
    }

    @Test
    public void testTheInplaceVariantsReturnTheirArgument() {
        MatrixD A = Matrices.randomUniformD(7, 3, -1.0, 1.0, 13L);
        assertSame("MatrixD", A, Statistics.shuffleRowsInplace(A));
        assertSame("MatrixD seeded", A, Statistics.shuffleRowsInplace(A, 1L));
        MatrixF F = Matrices.randomUniformF(7, 3, -1.0f, 1.0f, 13L);
        assertSame("MatrixF", F, Statistics.shuffleRowsInplace(F));
        assertSame("MatrixF seeded", F, Statistics.shuffleRowsInplace(F, 1L));
    }

    // the shuffle as it was before it became column-first: Fisher-Yates over
    // whole rows, each swap striding through the array by rows
    private static void rowSwappingShuffle(double[] a, int rows, int cols, XoShiRo256StarStar rnd) {
        for (int i = rows; i > 1; --i) {
            int sourceRow = rnd.nextInt(i);
            int targetRow = i - 1;
            if (sourceRow != targetRow) {
                for (int col = 0; col < cols; ++col) {
                    int t = col * rows + targetRow;
                    int s = col * rows + sourceRow;
                    double tmp = a[t];
                    a[t] = a[s];
                    a[s] = tmp;
                }
            }
        }
    }

    private static void rowSwappingShuffle(float[] a, int rows, int cols, XoShiRo256StarStar rnd) {
        for (int i = rows; i > 1; --i) {
            int sourceRow = rnd.nextInt(i);
            int targetRow = i - 1;
            if (sourceRow != targetRow) {
                for (int col = 0; col < cols; ++col) {
                    int t = col * rows + targetRow;
                    int s = col * rows + sourceRow;
                    float tmp = a[t];
                    a[t] = a[s];
                    a[s] = tmp;
                }
            }
        }
    }

    private static boolean sameRows(MatrixD a, MatrixD b) {
        return sameRows(a.getArrayUnsafe(), b.getArrayUnsafe(), a.numRows(), a.numColumns(), b.numRows(),
                b.numColumns());
    }

    private static boolean sameRows(MatrixF a, MatrixF b) {
        return sameRows(widen(a.getArrayUnsafe()), widen(b.getArrayUnsafe()), a.numRows(), a.numColumns(),
                b.numRows(), b.numColumns());
    }

    private static boolean sameRows(double[] a, double[] b, int rows, int cols, int bRows, int bCols) {
        if (rows != bRows || cols != bCols) {
            return false;
        }
        boolean[] used = new boolean[rows];
        for (int j = 0; j < rows; ++j) {
            double[] rowOfB = row(b, rows, cols, j);
            boolean found = false;
            for (int k = 0; k < rows && !found; ++k) {
                if (!used[k] && Arrays.equals(row(a, rows, cols, k), rowOfB)) {
                    used[k] = true;
                    found = true;
                }
            }
            if (!found) {
                return false;
            }
        }
        return true;
    }

    private static double[] row(double[] a, int rows, int cols, int row) {
        double[] x = new double[cols];
        for (int col = 0; col < cols; ++col) {
            x[col] = a[col * rows + row];
        }
        return x;
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

    private static double[] widen(float[] a) {
        double[] w = new double[a.length];
        for (int i = 0; i < a.length; ++i) {
            w[i] = a[i];
        }
        return w;
    }
}
