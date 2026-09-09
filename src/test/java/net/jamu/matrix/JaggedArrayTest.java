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
import static org.junit.Assert.fail;

import org.junit.Test;

/**
 * Tests for the four Matrices.fromJaggedArray factories.
 */
public final class JaggedArrayTest {

    @Test
    public void testAnEmptyOuterArrayIsRejected() {
        rejects("MatrixD", () -> Matrices.fromJaggedArrayD(new double[0][]));
        rejects("MatrixF", () -> Matrices.fromJaggedArrayF(new float[0][]));
        rejects("ComplexMatrixD", () -> Matrices.fromJaggedComplexArrayD(new double[0][]));
        rejects("ComplexMatrixF", () -> Matrices.fromJaggedComplexArrayF(new float[0][]));
    }

    @Test
    public void testANullRowIsRejected() {
        for (int nullAt : new int[] { 0, 1, 2 }) {
            String at = " with row " + nullAt + " null";
            rejects("MatrixD" + at, () -> Matrices.fromJaggedArrayD(dWithNullAt(nullAt)));
            rejects("MatrixF" + at, () -> Matrices.fromJaggedArrayF(fWithNullAt(nullAt)));
            rejects("ComplexMatrixD" + at, () -> Matrices.fromJaggedComplexArrayD(dWithNullAt(nullAt)));
            rejects("ComplexMatrixF" + at, () -> Matrices.fromJaggedComplexArrayF(fWithNullAt(nullAt)));
        }
    }

    @Test
    public void testANullArrayThrowsNullPointerException() {
        npe("MatrixD", () -> Matrices.fromJaggedArrayD(null));
        npe("MatrixF", () -> Matrices.fromJaggedArrayF(null));
        npe("ComplexMatrixD", () -> Matrices.fromJaggedComplexArrayD(null));
        npe("ComplexMatrixF", () -> Matrices.fromJaggedComplexArrayF(null));
    }

    @Test
    public void testInconsistentRowLengthsAreRejected() {
        double[][] d = { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0 } };
        float[][] f = { { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f } };
        rejects("MatrixD", () -> Matrices.fromJaggedArrayD(d));
        rejects("MatrixF", () -> Matrices.fromJaggedArrayF(f));
        double[][] cd = { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0 } };
        float[][] cf = { { 1.0f, 2.0f, 3.0f, 4.0f }, { 5.0f, 6.0f } };
        rejects("ComplexMatrixD", () -> Matrices.fromJaggedComplexArrayD(cd));
        rejects("ComplexMatrixF", () -> Matrices.fromJaggedComplexArrayF(cf));
    }

    @Test
    public void testAZeroLengthRowIsRejected() {
        rejects("MatrixD", () -> Matrices.fromJaggedArrayD(new double[1][0]));
        rejects("MatrixF", () -> Matrices.fromJaggedArrayF(new float[1][0]));
        rejects("ComplexMatrixD", () -> Matrices.fromJaggedComplexArrayD(new double[1][0]));
        rejects("ComplexMatrixF", () -> Matrices.fromJaggedComplexArrayF(new float[1][0]));
    }

    @Test
    public void testAnOddRowLengthIsRejectedForComplex() {
        double[][] cd = { { 1.0, 2.0, 3.0 } };
        float[][] cf = { { 1.0f, 2.0f, 3.0f } };
        rejects("ComplexMatrixD", () -> Matrices.fromJaggedComplexArrayD(cd));
        rejects("ComplexMatrixF", () -> Matrices.fromJaggedComplexArrayF(cf));
    }

    @Test
    public void testTheHappyPathKeepsShapeAndContent() {
        double[][] d = { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
        MatrixD md = Matrices.fromJaggedArrayD(d);
        assertEquals("rows", 2, md.numRows());
        assertEquals("cols", 3, md.numColumns());
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 3; ++col) {
                assertEquals("MatrixD at " + row + "," + col, d[row][col], md.get(row, col), 0.0);
            }
        }

        float[][] f = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } };
        MatrixF mf = Matrices.fromJaggedArrayF(f);
        assertEquals("float rows", 2, mf.numRows());
        assertEquals("float cols", 3, mf.numColumns());

        // one complex number per pair, so this is a 2 x 2 matrix
        double[][] cd = { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 } };
        ComplexMatrixD mcd = Matrices.fromJaggedComplexArrayD(cd);
        assertEquals("complex rows", 2, mcd.numRows());
        assertEquals("complex cols", 2, mcd.numColumns());
        assertEquals("complex (0,0) real", 1.0, mcd.get(0, 0).re(), 0.0);
        assertEquals("complex (0,0) imag", 2.0, mcd.get(0, 0).im(), 0.0);
        assertEquals("complex (1,1) real", 7.0, mcd.get(1, 1).re(), 0.0);
        assertEquals("complex (1,1) imag", 8.0, mcd.get(1, 1).im(), 0.0);

        float[][] cf = { { 1.0f, 2.0f, 3.0f, 4.0f }, { 5.0f, 6.0f, 7.0f, 8.0f } };
        ComplexMatrixF mcf = Matrices.fromJaggedComplexArrayF(cf);
        assertEquals("complex float rows", 2, mcf.numRows());
        assertEquals("complex float cols", 2, mcf.numColumns());
    }

    @Test
    public void testTheRaggedDeclarationFormWorks() {
        // rows assigned one by one, which is how the outer array is often built
        double[][] d = new double[3][];
        d[0] = new double[] { 8.0, 8.0, 8.0 };
        d[1] = new double[] { 7.0, 7.0, 7.0 };
        d[2] = new double[] { 9.0, 9.0, 9.0 };
        MatrixD m = Matrices.fromJaggedArrayD(d);
        assertEquals("rows", 3, m.numRows());
        assertEquals("cols", 3, m.numColumns());
        assertEquals("(1,2)", 7.0, m.get(1, 2), 0.0);
        assertEquals("(2,0)", 9.0, m.get(2, 0), 0.0);
    }

    @Test
    public void testTheComplexRoundTripThroughToJaggedArray() {
        ComplexMatrixD cd = Matrices.randomUniformComplexD(4, 3, -1.0, 1.0, 5L);
        ComplexMatrixD back = Matrices.fromJaggedComplexArrayD(cd.toJaggedArray());
        assertEquals("rows", cd.numRows(), back.numRows());
        assertEquals("cols", cd.numColumns(), back.numColumns());
        double[] want = cd.getArrayUnsafe();
        double[] got = back.getArrayUnsafe();
        assertEquals("length", want.length, got.length);
        for (int i = 0; i < want.length; ++i) {
            assertEquals("at " + i, want[i], got[i], 0.0);
        }
    }

    private static double[][] dWithNullAt(int idx) {
        double[][] d = new double[3][];
        for (int i = 0; i < 3; ++i) {
            d[i] = (i == idx) ? null : new double[] { 1.0, 2.0 };
        }
        return d;
    }

    private static float[][] fWithNullAt(int idx) {
        float[][] f = new float[3][];
        for (int i = 0; i < 3; ++i) {
            f[i] = (i == idx) ? null : new float[] { 1.0f, 2.0f };
        }
        return f;
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

    private static void npe(String what, Body body) {
        try {
            body.run();
            fail(what + " : expected a NullPointerException but none was thrown");
        } catch (NullPointerException expected) {
            // null is not a matrix
        }
    }
}
