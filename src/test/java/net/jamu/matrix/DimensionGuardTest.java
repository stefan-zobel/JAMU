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
import static org.junit.Assert.fail;

import org.junit.Test;

/**
 * Tests for the dimension guards behind reshape and matrix construction.
 */
public final class DimensionGuardTest {

    // 4 * 1073741825 is 2^32 + 4, which truncates to 4 in an int
    private static final int OVER_ROWS_4 = 1073741825;
    private static final int OVER_COLS_4 = 4;
    // 3 * 1431655766 is 2^32 + 2, which truncates to 2 in an int
    private static final int OVER_ROWS_2 = 3;
    private static final int OVER_COLS_2 = 1431655766;

    private static final int[][] VALID_SHAPES = { { 4, 6 }, { 8, 3 }, { 3, 8 }, { 2, 12 }, { 12, 2 },
            { 24, 1 }, { 1, 24 }, { 6, 4 } };

    @Test
    public void testReshapeRejectsAProductThatOverflowsAnInt() {
        rejects("MatrixD 2x2", () -> Matrices.createD(2, 2).reshape(OVER_ROWS_4, OVER_COLS_4));
        rejects("MatrixF 2x2", () -> Matrices.createF(2, 2).reshape(OVER_ROWS_4, OVER_COLS_4));
        rejects("ComplexMatrixD 2x2",
                () -> Matrices.createComplexD(2, 2).reshape(OVER_ROWS_4, OVER_COLS_4));
        rejects("ComplexMatrixF 2x2",
                () -> Matrices.createComplexF(2, 2).reshape(OVER_ROWS_4, OVER_COLS_4));

        rejects("MatrixD 1x2", () -> Matrices.createD(1, 2).reshape(OVER_ROWS_2, OVER_COLS_2));
        rejects("MatrixF 1x2", () -> Matrices.createF(1, 2).reshape(OVER_ROWS_2, OVER_COLS_2));
        rejects("ComplexMatrixD 1x2",
                () -> Matrices.createComplexD(1, 2).reshape(OVER_ROWS_2, OVER_COLS_2));
        rejects("ComplexMatrixF 1x2",
                () -> Matrices.createComplexF(1, 2).reshape(OVER_ROWS_2, OVER_COLS_2));
    }

    @Test
    public void testReshapeAcceptsEveryValidShape() {
        MatrixD d = Matrices.randomUniformD(6, 4, 5L);
        double[] wantD = d.getArrayUnsafe().clone();
        MatrixF f = Matrices.randomUniformF(6, 4, 5L);
        float[] wantF = f.getArrayUnsafe().clone();
        ComplexMatrixD cd = Matrices.randomUniformComplexD(6, 4, 5L);
        double[] wantCD = cd.getArrayUnsafe().clone();
        ComplexMatrixF cf = Matrices.randomUniformComplexF(6, 4, 5L);
        float[] wantCF = cf.getArrayUnsafe().clone();

        for (int[] s : VALID_SHAPES) {
            String at = " to " + s[0] + "x" + s[1];
            MatrixD rd = d.reshape(s[0], s[1]);
            assertEquals("MatrixD rows" + at, s[0], rd.numRows());
            assertEquals("MatrixD cols" + at, s[1], rd.numColumns());
            assertArrayEquals("MatrixD content" + at, wantD, rd.getArrayUnsafe(), 0.0);
            assertFalse("MatrixD must not alias" + at, rd.getArrayUnsafe() == d.getArrayUnsafe());

            MatrixF rf = f.reshape(s[0], s[1]);
            assertEquals("MatrixF rows" + at, s[0], rf.numRows());
            assertArrayEquals("MatrixF content" + at, wantF, rf.getArrayUnsafe(), 0.0f);

            ComplexMatrixD rcd = cd.reshape(s[0], s[1]);
            assertEquals("ComplexMatrixD rows" + at, s[0], rcd.numRows());
            assertEquals("ComplexMatrixD cols" + at, s[1], rcd.numColumns());
            assertArrayEquals("ComplexMatrixD content" + at, wantCD, rcd.getArrayUnsafe(), 0.0);

            ComplexMatrixF rcf = cf.reshape(s[0], s[1]);
            assertEquals("ComplexMatrixF rows" + at, s[0], rcf.numRows());
            assertArrayEquals("ComplexMatrixF content" + at, wantCF, rcf.getArrayUnsafe(), 0.0f);
        }
    }

    @Test
    public void testReshapeRejectsAnIncompatibleButRepresentableShape() {
        rejects("MatrixD 6x4 to 5x5", () -> Matrices.createD(6, 4).reshape(5, 5));
        rejects("MatrixD 6x4 to 6x5", () -> Matrices.createD(6, 4).reshape(6, 5));
        rejects("ComplexMatrixD 6x4 to 5x5", () -> Matrices.createComplexD(6, 4).reshape(5, 5));
    }

    @Test
    public void testReshapeRejectsNonPositiveDimensions() {
        rejects("zero rows", () -> Matrices.createD(6, 4).reshape(0, 4));
        rejects("zero cols", () -> Matrices.createD(6, 4).reshape(6, 0));
        rejects("negative rows", () -> Matrices.createD(6, 4).reshape(-1, 4));
        rejects("negative cols", () -> Matrices.createD(6, 4).reshape(6, -1));
        rejects("complex zero rows", () -> Matrices.createComplexD(6, 4).reshape(0, 4));
        rejects("complex negative cols", () -> Matrices.createComplexD(6, 4).reshape(6, -1));
    }

    @Test
    public void testTheArrayLengthGuardRejectsAnOverflowingProduct() {
        rejects("SimpleMatrixD",
                () -> new SimpleMatrixD(OVER_ROWS_4, OVER_COLS_4, new double[4]));
        rejects("SimpleMatrixF",
                () -> new SimpleMatrixF(OVER_ROWS_4, OVER_COLS_4, new float[4]));
        rejects("SimpleComplexMatrixD",
                () -> new SimpleComplexMatrixD(OVER_ROWS_4, OVER_COLS_4, new double[8]));
        rejects("SimpleComplexMatrixF",
                () -> new SimpleComplexMatrixF(OVER_ROWS_4, OVER_COLS_4, new float[8]));

        rejects("SimpleMatrixD 2",
                () -> new SimpleMatrixD(OVER_ROWS_2, OVER_COLS_2, new double[2]));
        rejects("SimpleComplexMatrixD 2",
                () -> new SimpleComplexMatrixD(OVER_ROWS_2, OVER_COLS_2, new double[4]));
    }

    @Test
    public void testTheArrayLengthGuardAcceptsTheRightLength() {
        assertEquals("SimpleMatrixD", 6, new SimpleMatrixD(2, 3, new double[6]).getArrayUnsafe().length);
        assertEquals("SimpleMatrixF", 6, new SimpleMatrixF(2, 3, new float[6]).getArrayUnsafe().length);
        assertEquals("SimpleComplexMatrixD", 12,
                new SimpleComplexMatrixD(2, 3, new double[12]).getArrayUnsafe().length);
        assertEquals("SimpleComplexMatrixF", 12,
                new SimpleComplexMatrixF(2, 3, new float[12]).getArrayUnsafe().length);
        rejects("one element too few", () -> new SimpleMatrixD(2, 3, new double[5]));
        rejects("one element too many", () -> new SimpleMatrixD(2, 3, new double[7]));
        rejects("complex one too few", () -> new SimpleComplexMatrixD(2, 3, new double[11]));
    }


    // The two guards back each other up, so reshape throws even when only one
    // of them is intact. This one drives the compatibility guard on its own.
    @Test
    public void testTheCompatibilityGuardRejectsAnOverflowingProduct() {
        MatrixD a = Matrices.createD(2, 2);
        MatrixD b = Matrices.createD(1, 2);
        ComplexMatrixD ca = Matrices.createComplexD(2, 2);
        rejects("2x2 against 1073741825x4",
                () -> Checks.checkCompatibleDimension(a, OVER_ROWS_4, OVER_COLS_4));
        rejects("1x2 against 3x1431655766",
                () -> Checks.checkCompatibleDimension(b, OVER_ROWS_2, OVER_COLS_2));
        rejects("complex 2x2 against 1073741825x4",
                () -> Checks.checkCompatibleDimension(ca, OVER_ROWS_4, OVER_COLS_4));
    }

    @Test
    public void testTheCompatibilityGuardAcceptsEveryValidShape() {
        MatrixD a = Matrices.createD(6, 4);
        for (int[] s : VALID_SHAPES) {
            Checks.checkCompatibleDimension(a, s[0], s[1]);
        }
        rejects("6x4 against 5x5", () -> Checks.checkCompatibleDimension(a, 5, 5));
        rejects("6x4 against 4x5", () -> Checks.checkCompatibleDimension(a, 4, 5));
        rejects("6x4 against 6x5", () -> Checks.checkCompatibleDimension(a, 6, 5));
    }
    private interface Body {
        void run();
    }

    private static void rejects(String what, Body body) {
        try {
            body.run();
            fail(what + " : expected an IllegalArgumentException but none was thrown");
        } catch (IllegalArgumentException expected) {
            // the guard did its job
        }
    }
}
