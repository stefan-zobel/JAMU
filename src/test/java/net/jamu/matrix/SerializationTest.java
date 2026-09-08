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
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
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


    private static final int[] CHUNKS = { 1, 2, 3, 4, 5, 7, 8, 16, 1024 };
    // 0 and 1 stop inside the two marker bytes, 5 and 9 inside rows and cols,
    // 10 leaves the header complete with no element, 105 is one byte short
    private static final int[] KEEP = { 0, 1, 2, 5, 9, 10, 11, 50, 66, 105 };

    @Test
    public void testTheRoundTripSurvivesShortReads() throws IOException {
        MatrixD d = Matrices.randomUniformD(4, 3, -1.0, 1.0, 11L);
        MatrixF f = Matrices.randomUniformF(4, 3, -1.0f, 1.0f, 11L);
        ComplexMatrixD cd = Matrices.randomUniformComplexD(4, 3, -1.0, 1.0, 11L);
        ComplexMatrixF cf = Matrices.randomUniformComplexF(4, 3, -1.0f, 1.0f, 11L);
        for (int k : CHUNKS) {
            String at = " at chunk " + k;
            assertBitsD("MatrixD" + at, d.getArrayUnsafe(),
                    Matrices.deserializeD(new Choppy(bytes(d), k)).getArrayUnsafe());
            assertBitsF("MatrixF" + at, f.getArrayUnsafe(),
                    Matrices.deserializeF(new Choppy(bytes(f), k)).getArrayUnsafe());
            assertBitsD("ComplexMatrixD" + at, cd.getArrayUnsafe(),
                    Matrices.deserializeComplexD(new Choppy(bytes(cd), k)).getArrayUnsafe());
            assertBitsF("ComplexMatrixF" + at, cf.getArrayUnsafe(),
                    Matrices.deserializeComplexF(new Choppy(bytes(cf), k)).getArrayUnsafe());
        }
    }

    @Test
    public void testATruncatedStreamIsRejected() throws IOException {
        byte[] d = bytes(Matrices.randomUniformD(4, 3, -1.0, 1.0, 11L));
        assertEquals("the fixture must be 106 bytes", 106, d.length);
        for (int keep : KEEP) {
            atEof("MatrixD truncated to " + keep, () -> Matrices.deserializeD(in(cut(d, keep))));
        }
        byte[] f = bytes(Matrices.randomUniformF(4, 3, -1.0f, 1.0f, 11L));
        atEof("MatrixF truncated to 30", () -> Matrices.deserializeF(in(cut(f, 30))));
        byte[] cd = bytes(Matrices.randomUniformComplexD(4, 3, -1.0, 1.0, 11L));
        atEof("ComplexMatrixD truncated to 30", () -> Matrices.deserializeComplexD(in(cut(cd, 30))));
        byte[] cf = bytes(Matrices.randomUniformComplexF(4, 3, -1.0f, 1.0f, 11L));
        atEof("ComplexMatrixF truncated to 30", () -> Matrices.deserializeComplexF(in(cut(cf, 30))));
    }

    @Test
    public void testATruncatedFileIsRejected() throws IOException {
        Path p = Files.createTempFile("jamu", ".bin");
        try {
            MatrixD d = Matrices.randomUniformD(40, 30, -1.0, 1.0, 9L);
            Matrices.serializeD(d, p);
            byte[] all = Files.readAllBytes(p);
            Files.write(p, cut(all, all.length - 1));
            atEof("a file one byte short", () -> Matrices.deserializeD(p));
        } finally {
            Files.deleteIfExists(p);
        }
    }

    private static byte[] cut(byte[] all, int keep) {
        byte[] c = new byte[keep];
        System.arraycopy(all, 0, c, 0, keep);
        return c;
    }

    private static void atEof(String what, Body body) {
        try {
            body.run();
            fail(what + " : expected an EOFException but none was thrown");
        } catch (EOFException expected) {
            // the reader noticed that the stream ran out
        } catch (IOException other) {
            fail(what + " : expected an EOFException but got " + other);
        }
    }

    /** a legal InputStream that hands out at most n bytes per read */
    private static final class Choppy extends InputStream {
        private final byte[] b;
        private final int n;
        private int pos;

        Choppy(byte[] b, int n) {
            this.b = b;
            this.n = n;
        }

        @Override
        public int read() {
            return (pos < b.length) ? (b[pos++] & 0xFF) : -1;
        }

        @Override
        public int read(byte[] dst, int off, int len) {
            if (pos >= b.length) {
                return -1;
            }
            int k = Math.min(Math.min(len, n), b.length - pos);
            System.arraycopy(b, pos, dst, off, k);
            pos += k;
            return k;
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
