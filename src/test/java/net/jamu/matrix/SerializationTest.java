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
import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.Test;

/**
 * Tests for the binary matrix format written by Matrices.serialize and read
 * back by Matrices.deserialize.
 */
public final class SerializationTest {

    private static final int HEADER = 10;
    private static final int[][] SHAPES = { { 1, 1 }, { 4, 3 }, { 3, 4 }, { 200, 150 } };

    @Test
    public void testTheRoundTripIsBitExact() throws IOException {
        for (int[] s : SHAPES) {
            String at = " at " + s[0] + "x" + s[1];
            MatrixD d = Matrices.randomUniformD(s[0], s[1], -1.0, 1.0, 5L);
            assertBitsD("MatrixD" + at, d.getArrayUnsafe(), roundTrip(d).getArrayUnsafe());
            MatrixF f = Matrices.randomUniformF(s[0], s[1], -1.0f, 1.0f, 5L);
            assertBitsF("MatrixF" + at, f.getArrayUnsafe(), roundTrip(f).getArrayUnsafe());
            ComplexMatrixD cd = Matrices.randomUniformComplexD(s[0], s[1], -1.0, 1.0, 5L);
            assertBitsD("ComplexMatrixD" + at, cd.getArrayUnsafe(), roundTrip(cd).getArrayUnsafe());
            ComplexMatrixF cf = Matrices.randomUniformComplexF(s[0], s[1], -1.0f, 1.0f, 5L);
            assertBitsF("ComplexMatrixF" + at, cf.getArrayUnsafe(), roundTrip(cf).getArrayUnsafe());
        }
    }

    @Test
    public void testTheReportedSizeIsTheWrittenSize() throws IOException {
        for (int[] s : SHAPES) {
            int n = s[0] * s[1];
            String at = " at " + s[0] + "x" + s[1];
            check("MatrixD" + at, Matrices.randomUniformD(s[0], s[1], 5L), HEADER + 8 * n);
            check("MatrixF" + at, Matrices.randomUniformF(s[0], s[1], 5L), HEADER + 4 * n);
            check("ComplexMatrixD" + at, Matrices.randomUniformComplexD(s[0], s[1], 5L),
                    HEADER + 16 * n);
            check("ComplexMatrixF" + at, Matrices.randomUniformComplexF(s[0], s[1], 5L),
                    HEADER + 8 * n);
        }
    }

    @Test
    public void testTheHeaderLayout() throws IOException {
        // endian marker, type byte, then rows and cols big endian
        assertArrayEquals("MatrixD", new byte[] { 1, 0x40, 0, 0, 0, 4, 0, 0, 0, 3 },
                header(Matrices.createD(4, 3)));
        assertArrayEquals("MatrixF", new byte[] { 1, 0x20, 0, 0, 0, 4, 0, 0, 0, 3 },
                header(Matrices.createF(4, 3)));
        assertArrayEquals("ComplexMatrixD", new byte[] { 1, (byte) 0xC0, 0, 0, 0, 4, 0, 0, 0, 3 },
                header(Matrices.createComplexD(4, 3)));
        assertArrayEquals("ComplexMatrixF", new byte[] { 1, (byte) 0xE0, 0, 0, 0, 4, 0, 0, 0, 3 },
                header(Matrices.createComplexF(4, 3)));
        assertArrayEquals("a shape that is not square", new byte[] { 1, 0x40, 0, 0, 0, 1, 0, 0, 0, 7 },
                header(Matrices.createD(1, 7)));
    }

    @Test
    public void testTheFloatBytesAreUnchanged() throws IOException {
        // pins backward compatibility: the float format must not move
        MatrixF m = Matrices.createF(2, 2);
        m.set(0, 0, 1.0f);
        m.set(1, 0, 2.0f);
        m.set(0, 1, -0.5f);
        m.set(1, 1, 0.0f);
        byte[] expected = { 1, 0x20, 0, 0, 0, 2, 0, 0, 0, 2, 0x3F, (byte) 0x80, 0, 0, 0x40, 0, 0, 0,
                (byte) 0xBF, 0, 0, 0, 0, 0, 0, 0 };
        assertArrayEquals("MatrixF bytes", expected, bytes(m));
    }

    @Test
    public void testSpecialValuesSurviveTheRoundTrip() throws IOException {
        double[] special = { 0.0, -0.0, Double.NaN, Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY, Double.MIN_VALUE, Double.MAX_VALUE, -1.0 };
        MatrixD d = Matrices.createD(4, 2);
        for (int col = 0, k = 0; col < 2; ++col) {
            for (int row = 0; row < 4; ++row, ++k) {
                d.set(row, col, special[k]);
            }
        }
        assertBitsD("MatrixD special values", d.getArrayUnsafe(), roundTrip(d).getArrayUnsafe());

        float[] specialF = { 0.0f, -0.0f, Float.NaN, Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.MIN_VALUE, Float.MAX_VALUE, -1.0f };
        MatrixF f = Matrices.createF(4, 2);
        for (int col = 0, k = 0; col < 2; ++col) {
            for (int row = 0; row < 4; ++row, ++k) {
                f.set(row, col, specialF[k]);
            }
        }
        assertBitsF("MatrixF special values", f.getArrayUnsafe(), roundTrip(f).getArrayUnsafe());
    }

    @Test
    public void testTheRoundTripThroughAFile() throws IOException {
        Path p = Files.createTempFile("jamu", ".bin");
        try {
            MatrixD d = Matrices.randomUniformD(40, 30, -1.0, 1.0, 9L);
            long sz = Matrices.serializeD(d, p);
            assertEquals("reported size", Files.size(p), sz);
            assertEquals("expected size", HEADER + 8 * 40 * 30, sz);
            assertBitsD("MatrixD", d.getArrayUnsafe(), Matrices.deserializeD(p).getArrayUnsafe());

            ComplexMatrixD cd = Matrices.randomUniformComplexD(40, 30, -1.0, 1.0, 9L);
            long csz = Matrices.serializeComplexD(cd, p);
            assertEquals("complex reported size", Files.size(p), csz);
            assertBitsD("ComplexMatrixD", cd.getArrayUnsafe(),
                    Matrices.deserializeComplexD(p).getArrayUnsafe());

            MatrixF f = Matrices.randomUniformF(40, 30, -1.0f, 1.0f, 9L);
            long fsz = Matrices.serializeF(f, p);
            assertEquals("float reported size", Files.size(p), fsz);
            assertBitsF("MatrixF", f.getArrayUnsafe(), Matrices.deserializeF(p).getArrayUnsafe());
        } finally {
            Files.deleteIfExists(p);
        }
    }

    @Test
    public void testAStreamIsRejectedByTheWrongDeserializer() throws IOException {
        byte[] d = bytes(Matrices.randomUniformD(3, 2, 5L));
        byte[] f = bytes(Matrices.randomUniformF(3, 2, 5L));
        byte[] cd = bytes(Matrices.randomUniformComplexD(3, 2, 5L));
        byte[] cf = bytes(Matrices.randomUniformComplexF(3, 2, 5L));

        rejects("double read as float", () -> Matrices.deserializeF(in(d)));
        rejects("double read as complex double", () -> Matrices.deserializeComplexD(in(d)));
        rejects("float read as double", () -> Matrices.deserializeD(in(f)));
        rejects("float read as complex float", () -> Matrices.deserializeComplexF(in(f)));
        rejects("complex double read as double", () -> Matrices.deserializeD(in(cd)));
        rejects("complex double read as complex float", () -> Matrices.deserializeComplexF(in(cd)));
        rejects("complex float read as float", () -> Matrices.deserializeF(in(cf)));
        rejects("complex float read as complex double", () -> Matrices.deserializeComplexD(in(cf)));
    }

    private static ByteArrayInputStream in(byte[] b) {
        return new ByteArrayInputStream(b);
    }

    private static byte[] bytes(MatrixD m) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Matrices.serializeD(m, bos);
        return bos.toByteArray();
    }

    private static byte[] bytes(MatrixF m) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Matrices.serializeF(m, bos);
        return bos.toByteArray();
    }

    private static byte[] bytes(ComplexMatrixD m) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Matrices.serializeComplexD(m, bos);
        return bos.toByteArray();
    }

    private static byte[] bytes(ComplexMatrixF m) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Matrices.serializeComplexF(m, bos);
        return bos.toByteArray();
    }

    private static byte[] header(MatrixD m) throws IOException {
        return first(bytes(m));
    }

    private static byte[] header(MatrixF m) throws IOException {
        return first(bytes(m));
    }

    private static byte[] header(ComplexMatrixD m) throws IOException {
        return first(bytes(m));
    }

    private static byte[] header(ComplexMatrixF m) throws IOException {
        return first(bytes(m));
    }

    private static byte[] first(byte[] all) {
        byte[] h = new byte[HEADER];
        System.arraycopy(all, 0, h, 0, HEADER);
        return h;
    }

    private static MatrixD roundTrip(MatrixD m) throws IOException {
        return Matrices.deserializeD(in(bytes(m)));
    }

    private static MatrixF roundTrip(MatrixF m) throws IOException {
        return Matrices.deserializeF(in(bytes(m)));
    }

    private static ComplexMatrixD roundTrip(ComplexMatrixD m) throws IOException {
        return Matrices.deserializeComplexD(in(bytes(m)));
    }

    private static ComplexMatrixF roundTrip(ComplexMatrixF m) throws IOException {
        return Matrices.deserializeComplexF(in(bytes(m)));
    }

    private static void check(String what, MatrixD m, int expected) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        long sz = Matrices.serializeD(m, bos);
        assertEquals(what + " reported", expected, sz);
        assertEquals(what + " written", expected, bos.size());
    }

    private static void check(String what, MatrixF m, int expected) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        long sz = Matrices.serializeF(m, bos);
        assertEquals(what + " reported", expected, sz);
        assertEquals(what + " written", expected, bos.size());
    }

    private static void check(String what, ComplexMatrixD m, int expected) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        long sz = Matrices.serializeComplexD(m, bos);
        assertEquals(what + " reported", expected, sz);
        assertEquals(what + " written", expected, bos.size());
    }

    private static void check(String what, ComplexMatrixF m, int expected) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        long sz = Matrices.serializeComplexF(m, bos);
        assertEquals(what + " reported", expected, sz);
        assertEquals(what + " written", expected, bos.size());
    }

    private static void assertBitsD(String what, double[] expected, double[] actual) {
        assertEquals(what + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            long e = Double.doubleToRawLongBits(expected[i]);
            long a = Double.doubleToRawLongBits(actual[i]);
            if (e != a) {
                fail(what + " differs at " + i + " : " + expected[i] + " became " + actual[i]);
            }
        }
    }

    private static void assertBitsF(String what, float[] expected, float[] actual) {
        assertEquals(what + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            int e = Float.floatToRawIntBits(expected[i]);
            int a = Float.floatToRawIntBits(actual[i]);
            if (e != a) {
                fail(what + " differs at " + i + " : " + expected[i] + " became " + actual[i]);
            }
        }
    }

    private interface Body {
        void run() throws IOException;
    }

    private static void rejects(String what, Body body) {
        try {
            body.run();
            fail(what + " : expected an IOException but none was thrown");
        } catch (IOException expected) {
            // the type check did its job
        }
    }
}
