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
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

/**
 * Tests for {@code Statistics.shuffleColumns} and
 * {@code Statistics.shuffleColumnsInplace}. The seeded tests pin the
 * permutation a seed produces to a plain Fisher-Yates over whole columns, and
 * the seeded copying variant to the seeded in-place variant.
 */
public final class ShuffleColumnsTest {

    private static final int[][] SHAPES = { { 1, 1 }, { 1, 7 }, { 7, 1 }, { 2, 2 }, { 7, 3 }, { 3, 7 },
            { 20, 15 }, { 50, 200 } };

    private static final long[] SEEDS = { 0L, 1L, 42L, -7L, 123456789L };

    @Test
    public void testMatrixDSeededShuffleMatchesAColumnSwappingFisherYates() {
        for (int[] s : SHAPES) {
            for (long seed : SEEDS) {
                String at = " at " + s[0] + "x" + s[1] + " seed " + seed;
                MatrixD A = Matrices.randomUniformD(s[0], s[1], -1.0, 1.0, 5L);
                double[] expected = A.getArrayUnsafe().clone();
                columnSwappingShuffle(expected, s[0], s[1], new XoShiRo256StarStar(seed));
                assertBits("shuffleColumns(MatrixD, seed)" + at, expected,
                        Statistics.shuffleColumns(A, seed).getArrayUnsafe());
                assertBits("shuffleColumnsInplace(MatrixD, seed)" + at, expected,
                        Statistics.shuffleColumnsInplace(A, seed).getArrayUnsafe());
            }
        }
    }

    @Test
    public void testMatrixFSeededShuffleMatchesAColumnSwappingFisherYates() {
        for (int[] s : SHAPES) {
            for (long seed : SEEDS) {
                String at = " at " + s[0] + "x" + s[1] + " seed " + seed;
                MatrixF A = Matrices.randomUniformF(s[0], s[1], -1.0f, 1.0f, 5L);
                float[] expected = A.getArrayUnsafe().clone();
                columnSwappingShuffle(expected, s[0], s[1], new XoShiRo256StarStar(seed));
                assertBits("shuffleColumns(MatrixF, seed)" + at, expected,
                        Statistics.shuffleColumns(A, seed).getArrayUnsafe());
                assertBits("shuffleColumnsInplace(MatrixF, seed)" + at, expected,
                        Statistics.shuffleColumnsInplace(A, seed).getArrayUnsafe());
            }
        }
    }

    @Test
    public void testTheSameSeedGivesTheSameShuffle() {
        MatrixD A = Matrices.randomUniformD(50, 200, -1.0, 1.0, 5L);
        assertBits("MatrixD", Statistics.shuffleColumns(A, 42L).getArrayUnsafe(),
                Statistics.shuffleColumns(A, 42L).getArrayUnsafe());
        MatrixF F = Matrices.randomUniformF(50, 200, -1.0f, 1.0f, 5L);
        assertBits("MatrixF", Statistics.shuffleColumns(F, 42L).getArrayUnsafe(),
                Statistics.shuffleColumns(F, 42L).getArrayUnsafe());
    }

    @Test
    public void testTheSeededShuffleActuallyMovesColumns() {
        MatrixD A = Matrices.randomUniformD(50, 200, -1.0, 1.0, 5L);
        assertFalse("MatrixD", Arrays.equals(A.getArrayUnsafe(), Statistics.shuffleColumns(A, 42L).getArrayUnsafe()));
        MatrixF F = Matrices.randomUniformF(50, 200, -1.0f, 1.0f, 5L);
        assertFalse("MatrixF", Arrays.equals(F.getArrayUnsafe(), Statistics.shuffleColumns(F, 42L).getArrayUnsafe()));
    }

    @Test
    public void testEveryVariantIsAPermutationOfTheColumns() {
        for (int[] s : SHAPES) {
            String at = " at " + s[0] + "x" + s[1];
            MatrixD A = Matrices.randomUniformD(s[0], s[1], -1.0, 1.0, 9L);
            assertTrue("shuffleColumns(MatrixD)" + at, sameColumns(A, Statistics.shuffleColumns(A)));
            assertTrue("shuffleColumns(MatrixD, seed)" + at, sameColumns(A, Statistics.shuffleColumns(A, 3L)));
            assertTrue("shuffleColumnsInplace(MatrixD)" + at,
                    sameColumns(A, Statistics.shuffleColumnsInplace(A.copy())));
            assertTrue("shuffleColumnsInplace(MatrixD, seed)" + at,
                    sameColumns(A, Statistics.shuffleColumnsInplace(A.copy(), 3L)));
            MatrixF F = Matrices.randomUniformF(s[0], s[1], -1.0f, 1.0f, 9L);
            assertTrue("shuffleColumns(MatrixF)" + at, sameColumns(F, Statistics.shuffleColumns(F)));
            assertTrue("shuffleColumns(MatrixF, seed)" + at, sameColumns(F, Statistics.shuffleColumns(F, 3L)));
            assertTrue("shuffleColumnsInplace(MatrixF)" + at,
                    sameColumns(F, Statistics.shuffleColumnsInplace(F.copy())));
            assertTrue("shuffleColumnsInplace(MatrixF, seed)" + at,
                    sameColumns(F, Statistics.shuffleColumnsInplace(F.copy(), 3L)));
        }
    }

    @Test
    public void testTheCopyingVariantsLeaveTheArgumentUntouched() {
        MatrixD A = Matrices.randomUniformD(15, 20, -1.0, 1.0, 11L);
        double[] before = A.getArrayUnsafe().clone();
        assertNotSame("MatrixD", A, Statistics.shuffleColumns(A));
        assertArrayEquals("MatrixD", before, A.getArrayUnsafe(), 0.0);
        assertNotSame("MatrixD seeded", A, Statistics.shuffleColumns(A, 42L));
        assertArrayEquals("MatrixD seeded", before, A.getArrayUnsafe(), 0.0);
        MatrixF F = Matrices.randomUniformF(15, 20, -1.0f, 1.0f, 11L);
        float[] fBefore = F.getArrayUnsafe().clone();
        assertNotSame("MatrixF", F, Statistics.shuffleColumns(F));
        assertArrayEquals("MatrixF", fBefore, F.getArrayUnsafe(), 0.0f);
        assertNotSame("MatrixF seeded", F, Statistics.shuffleColumns(F, 42L));
        assertArrayEquals("MatrixF seeded", fBefore, F.getArrayUnsafe(), 0.0f);
    }

    @Test
    public void testTheInplaceVariantsReturnTheirArgument() {
        MatrixD A = Matrices.randomUniformD(3, 7, -1.0, 1.0, 13L);
        assertSame("MatrixD", A, Statistics.shuffleColumnsInplace(A));
        assertSame("MatrixD seeded", A, Statistics.shuffleColumnsInplace(A, 1L));
        MatrixF F = Matrices.randomUniformF(3, 7, -1.0f, 1.0f, 13L);
        assertSame("MatrixF", F, Statistics.shuffleColumnsInplace(F));
        assertSame("MatrixF seeded", F, Statistics.shuffleColumnsInplace(F, 1L));
    }

    // Fisher-Yates over whole columns, element by element
    private static void columnSwappingShuffle(double[] a, int rows, int cols, XoShiRo256StarStar rnd) {
        for (int i = cols; i > 1; --i) {
            int sourceCol = rnd.nextInt(i);
            int targetCol = i - 1;
            if (sourceCol != targetCol) {
                for (int row = 0; row < rows; ++row) {
                    int t = targetCol * rows + row;
                    int s = sourceCol * rows + row;
                    double tmp = a[t];
                    a[t] = a[s];
                    a[s] = tmp;
                }
            }
        }
    }

    private static void columnSwappingShuffle(float[] a, int rows, int cols, XoShiRo256StarStar rnd) {
        for (int i = cols; i > 1; --i) {
            int sourceCol = rnd.nextInt(i);
            int targetCol = i - 1;
            if (sourceCol != targetCol) {
                for (int row = 0; row < rows; ++row) {
                    int t = targetCol * rows + row;
                    int s = sourceCol * rows + row;
                    float tmp = a[t];
                    a[t] = a[s];
                    a[s] = tmp;
                }
            }
        }
    }

    private static boolean sameColumns(MatrixD a, MatrixD b) {
        return sameColumns(a.getArrayUnsafe(), b.getArrayUnsafe(), a.numRows(), a.numColumns(), b.numRows(),
                b.numColumns());
    }

    private static boolean sameColumns(MatrixF a, MatrixF b) {
        return sameColumns(widen(a.getArrayUnsafe()), widen(b.getArrayUnsafe()), a.numRows(), a.numColumns(),
                b.numRows(), b.numColumns());
    }

    private static boolean sameColumns(double[] a, double[] b, int rows, int cols, int bRows, int bCols) {
        if (rows != bRows || cols != bCols) {
            return false;
        }
        boolean[] used = new boolean[cols];
        for (int j = 0; j < cols; ++j) {
            double[] colOfB = Arrays.copyOfRange(b, j * rows, (j + 1) * rows);
            boolean found = false;
            for (int k = 0; k < cols && !found; ++k) {
                if (!used[k] && Arrays.equals(Arrays.copyOfRange(a, k * rows, (k + 1) * rows), colOfB)) {
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
