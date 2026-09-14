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
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Executable;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

import org.junit.Test;

/**
 * Tests for a {@code MatrixF} view passed as an argument.
 */
public final class MatrixFViewArgumentTest {

    private static final long SEED = 20260915L;

    @Test
    public void testArrayOfAnOrdinaryMatrixIsItsBackingArray() {
        MatrixF A = random(7, 4, SEED);
        assertSame(A.getArrayUnsafe(), ReadAccess.array(A));
    }

    @Test
    public void testArrayOfAViewIsAFreshCopyOfTheRegion() {
        MatrixF P = random(9, 8, SEED);
        float[] before = P.getArrayUnsafe().clone();
        MatrixF V = Matrices.view(P, 2, 1, 6, 5);
        float[] first = ReadAccess.array(V);
        float[] second = ReadAccess.array(V);
        assertNotSame(P.getArrayUnsafe(), first);
        assertNotSame(first, second);
        assertBitsArray("first", V.copy().getArrayUnsafe(), first);
        assertBitsArray("second", V.copy().getArrayUnsafe(), second);
        Arrays.fill(first, 12345.0f);
        assertBitsArray("parent", before, P.getArrayUnsafe());
    }

    @Test
    public void testMatrixFBaseReadsAViewArgumentLikeItsCopy() {
        MatrixF P = random(60, 12, SEED + 1);
        float[] parent = P.getArrayUnsafe().clone();
        ArgCase[] cases = {
                new ArgCase("addInplace(B)", 6, 5, 6, 5, (T, B) -> T.addInplace(B)),
                new ArgCase("addInplace(alpha, B)", 6, 5, 6, 5, (T, B) -> T.addInplace(-1.5f, B)),
                new ArgCase("add(B, C)", 6, 5, 6, 5, (T, B) -> T.add(B, Matrices.createF(6, 5))),
                new ArgCase("add(alpha, B, C)", 6, 5, 6, 5, (T, B) -> T.add(2.5f, B, Matrices.createF(6, 5))),
                new ArgCase("plus(B)", 6, 5, 6, 5, (T, B) -> T.plus(B)),
                new ArgCase("addBroadcastedVectorInplace", 6, 5, 6, 1, (T, B) -> T.addBroadcastedVectorInplace(B)),
                new ArgCase("mulBroadcastedVectorInplace", 6, 5, 6, 1, (T, B) -> T.mulBroadcastedVectorInplace(B)),
                new ArgCase("mulBroadcastedVectorInplace full", 6, 5, 6, 5,
                        (T, B) -> T.mulBroadcastedVectorInplace(B)),
                new ArgCase("divBroadcastedVectorInplace", 6, 5, 6, 1, (T, B) -> T.divBroadcastedVectorInplace(B)),
                new ArgCase("divBroadcastedVectorInplace full", 6, 5, 6, 5,
                        (T, B) -> T.divBroadcastedVectorInplace(B)),
                new ArgCase("addBroadcastedRowVectorInplace", 6, 5, 1, 5,
                        (T, B) -> T.addBroadcastedRowVectorInplace(B)),
                new ArgCase("mulBroadcastedRowVectorInplace", 6, 5, 1, 5,
                        (T, B) -> T.mulBroadcastedRowVectorInplace(B)),
                new ArgCase("mulBroadcastedRowVectorInplace full", 6, 5, 6, 5,
                        (T, B) -> T.mulBroadcastedRowVectorInplace(B)),
                new ArgCase("divBroadcastedRowVectorInplace", 6, 5, 1, 5,
                        (T, B) -> T.divBroadcastedRowVectorInplace(B)),
                new ArgCase("divBroadcastedRowVectorInplace full", 6, 5, 6, 5,
                        (T, B) -> T.divBroadcastedRowVectorInplace(B)),
                new ArgCase("setInplace(B)", 6, 5, 6, 5, (T, B) -> T.setInplace(B)),
                new ArgCase("setInplace(alpha, B)", 6, 5, 6, 5, (T, B) -> T.setInplace(3.0f, B)),
                new ArgCase("setSubmatrixInplace height 10", 20, 8, 12, 6,
                        (T, B) -> T.setSubmatrixInplace(3, 1, B, 1, 1, 10, 5)),
                new ArgCase("setSubmatrixInplace height 45", 50, 8, 48, 6,
                        (T, B) -> T.setSubmatrixInplace(2, 1, B, 1, 1, 45, 5)),
                new ArgCase("setColumnInplace", 6, 5, 6, 1, (T, B) -> T.setColumnInplace(2, B)),
                new ArgCase("setInplaceUpperTrapezoidal", 6, 5, 7, 6, (T, B) -> T.setInplaceUpperTrapezoidal(B)),
                new ArgCase("setInplaceLowerTrapezoidal", 6, 5, 7, 6, (T, B) -> T.setInplaceLowerTrapezoidal(B)),
                new ArgCase("hadamard(B, out)", 6, 5, 6, 5, (T, B) -> T.hadamard(B, Matrices.createF(6, 5))),
                new ArgCase("hadamard(B)", 6, 5, 6, 5, (T, B) -> T.hadamard(B)),
                new ArgCase("appendColumn", 6, 5, 6, 1, (T, B) -> T.appendColumn(B)),
                new ArgCase("appendMatrix", 6, 5, 6, 3, (T, B) -> T.appendMatrix(B)),
                new ArgCase("hadamardTransposed", 6, 5, 5, 6, (T, B) -> T.hadamardTransposed(B)),
                new ArgCase("transposedHadamard", 6, 5, 5, 6, (T, B) -> T.transposedHadamard(B)) };
        Set<String> names = new HashSet<>();
        for (ArgCase c : cases) {
            assertEquals("duplicate " + c.name, true, names.add(c.name));
            MatrixF V = Matrices.view(P, 3, 2, 3 + c.argRows - 1, 2 + c.argCols - 1);
            MatrixF T1 = random(c.rows, c.cols, SEED + 2);
            MatrixF T2 = random(c.rows, c.cols, SEED + 2);
            Object got = c.call.apply(T1, V);
            Object want = c.call.apply(T2, V.copy());
            assertResult(c.name, want, got);
            assertBitsArray(c.name + ": target", T2.getArrayUnsafe(), T1.getArrayUnsafe());
            assertBitsArray(c.name + ": parent", parent, P.getArrayUnsafe());
        }
    }

    @Test
    public void testSelfOverlappingViewArgumentsMatchACopyTakenBefore() {
        Overlap[] cases = {
                new Overlap("setSubmatrixInplace down-right height 10", 50, 8, 0, 0, 9, 5,
                        (A, B) -> A.setSubmatrixInplace(1, 1, B, 0, 0, 9, 5)),
                new Overlap("setSubmatrixInplace up-left height 10", 50, 8, 1, 1, 10, 6,
                        (A, B) -> A.setSubmatrixInplace(0, 0, B, 0, 0, 9, 5)),
                new Overlap("setSubmatrixInplace inner region height 10", 50, 8, 0, 0, 20, 6,
                        (A, B) -> A.setSubmatrixInplace(2, 2, B, 1, 1, 10, 4)),
                new Overlap("setSubmatrixInplace down-right height 45", 50, 8, 0, 0, 44, 5,
                        (A, B) -> A.setSubmatrixInplace(1, 1, B, 0, 0, 44, 5)),
                new Overlap("setSubmatrixInplace up-left height 45", 50, 8, 1, 1, 45, 6,
                        (A, B) -> A.setSubmatrixInplace(0, 0, B, 0, 0, 44, 5)),
                new Overlap("setColumnInplace", 50, 8, 0, 1, 49, 1, (A, B) -> A.setColumnInplace(3, B)),
                new Overlap("addBroadcastedVectorInplace", 50, 8, 0, 2, 49, 2,
                        (A, B) -> A.addBroadcastedVectorInplace(B)),
                new Overlap("mulBroadcastedVectorInplace", 50, 8, 0, 2, 49, 2,
                        (A, B) -> A.mulBroadcastedVectorInplace(B)),
                new Overlap("divBroadcastedVectorInplace", 50, 8, 0, 2, 49, 2,
                        (A, B) -> A.divBroadcastedVectorInplace(B)),
                new Overlap("addBroadcastedRowVectorInplace", 50, 8, 3, 0, 3, 7,
                        (A, B) -> A.addBroadcastedRowVectorInplace(B)),
                new Overlap("mulBroadcastedRowVectorInplace", 50, 8, 3, 0, 3, 7,
                        (A, B) -> A.mulBroadcastedRowVectorInplace(B)),
                new Overlap("divBroadcastedRowVectorInplace", 50, 8, 3, 0, 3, 7,
                        (A, B) -> A.divBroadcastedRowVectorInplace(B)),
                new Overlap("addInplace", 50, 8, 0, 0, 49, 7, (A, B) -> A.addInplace(0.5f, B)),
                new Overlap("setInplace", 50, 8, 0, 0, 49, 7, (A, B) -> A.setInplace(-2.0f, B)),
                new Overlap("hadamard into the parent", 50, 8, 0, 0, 49, 7, (A, B) -> A.hadamard(B, A)),
                new Overlap("setInplaceUpperTrapezoidal", 50, 8, 2, 0, 49, 7,
                        (A, B) -> A.setInplaceUpperTrapezoidal(B)),
                new Overlap("setInplaceLowerTrapezoidal", 6, 20, 0, 3, 5, 19,
                        (A, B) -> A.setInplaceLowerTrapezoidal(B)) };
        for (Overlap c : cases) {
            MatrixF A1 = random(c.rows, c.cols, SEED + 3);
            MatrixF A2 = random(c.rows, c.cols, SEED + 3);
            MatrixF before = Matrices.view(A2, c.r0, c.c0, c.r1, c.c1).copy();
            c.call.accept(A1, Matrices.view(A1, c.r0, c.c0, c.r1, c.c1));
            c.call.accept(A2, before);
            assertBitsArray(c.name, A2.getArrayUnsafe(), A1.getArrayUnsafe());
        }
    }

    @Test
    public void testMatricesReadsAViewArgumentLikeItsCopy() {
        MatrixF P = random(20, 12, SEED + 4);
        P.set(17, 3, Float.NaN);
        float[] parent = P.getArrayUnsafe().clone();
        int[] v = { 3, 2, 8, 6 };
        int[] w = { 10, 4, 15, 8 };
        int[] row = { 5, 1, 5, 9 };
        int[] col = { 2, 7, 12, 7 };
        int[] nan = { 16, 2, 18, 4 };
        MatrixF O = random(6, 5, SEED + 5);
        MatrixF near = Matrices.view(P, v[0], v[1], v[2], v[3]).copy();
        near.set(4, 3, Math.nextUp(near.get(4, 3)));
        ComplexMatrixF Z = Matrices.randomUniformComplexF(4, 6, SEED + 6);
        Stat[] cases = {
                new Stat("serializeF(OutputStream)", true, v, v, (a, b) -> serialize(a)),
                new Stat("serializeF(Path)", true, v, v, (a, b) -> serializeToFile(a)),
                new Stat("convert", true, v, v, (a, b) -> Matrices.convert(a)),
                new Stat("convertToComplex", true, v, v, (a, b) -> Matrices.convertToComplex(a)),
                new Stat("ComplexMatrixF.times(MatrixF)", false, v, v, (a, b) -> Z.times(a)),
                new Stat("distance left", true, v, v, (a, b) -> Matrices.distance(a, O)),
                new Stat("distance right", true, v, v, (a, b) -> Matrices.distance(O, a)),
                new Stat("distance both", true, v, w, (a, b) -> Matrices.distance(a, b)),
                new Stat("approxEqual left", true, v, v, (a, b) -> Matrices.approxEqual(a, near)),
                new Stat("approxEqual right", true, v, v, (a, b) -> Matrices.approxEqual(near, a)),
                new Stat("approxEqual both", true, v, w, (a, b) -> Matrices.approxEqual(a, b)),
                new Stat("approxEqual relTol", true, v, v, (a, b) -> Matrices.approxEqual(a, near, 1.0e-17f)),
                new Stat("approxEqual absTol", true, v, v, (a, b) -> Matrices.approxEqual(near, a, 0.0f, 1.0e-12f)),
                new Stat("round", true, v, v, (a, b) -> Matrices.round(a, 3)),
                new Stat("round with NaN", true, nan, nan, (a, b) -> Matrices.round(a, 3)),
                new Stat("sumRows", true, v, v, (a, b) -> Matrices.sumRows(a)),
                new Stat("sumRows single row", true, row, row, (a, b) -> Matrices.sumRows(a)),
                new Stat("sumColumns", true, v, v, (a, b) -> Matrices.sumColumns(a)),
                new Stat("sumColumns single column", true, col, col, (a, b) -> Matrices.sumColumns(a)),
                new Stat("rowsAverage", true, v, v, (a, b) -> Matrices.rowsAverage(a)),
                new Stat("colsAverage", true, v, v, (a, b) -> Matrices.colsAverage(a)) };
        Set<String> names = new HashSet<>();
        for (Stat c : cases) {
            assertEquals("duplicate " + c.name, true, names.add(c.name));
            MatrixF a = Matrices.view(P, c.a[0], c.a[1], c.a[2], c.a[3]);
            MatrixF b = Matrices.view(P, c.b[0], c.b[1], c.b[2], c.b[3]);
            Object got = outcome(() -> c.call.apply(a, b));
            Object want = outcome(() -> c.call.apply(a.copy(), b.copy()));
            if (c.exact) {
                assertResult(c.name, want, got);
            } else {
                assertClose(c.name, (ComplexMatrixF) want, (ComplexMatrixF) got);
            }
            assertBitsArray(c.name + ": parent", parent, P.getArrayUnsafe());
        }
        assertEquals(IllegalArgumentException.class,
                outcome(() -> Matrices.round(Matrices.view(P, nan[0], nan[1], nan[2], nan[3]), 3)));
        assertEquals(true, Matrices.approxEqual(Matrices.view(P, v[0], v[1], v[2], v[3]), near));
        assertEquals(false, Matrices.approxEqual(Matrices.view(P, v[0], v[1], v[2], v[3]), near, 1.0e-17f));
    }

    @Test
    public void testTensorReadsAViewArgumentLikeItsCopy() {
        MatrixF P = random(15, 11, SEED + 7);
        float[] parent = P.getArrayUnsafe().clone();
        MatrixF V = Matrices.view(P, 4, 3, 9, 7);
        MatrixF C = V.copy();

        assertBitsArray("TensorF(MatrixF)", new TensorF(C).getArrayUnsafe(), new TensorF(V).getArrayUnsafe());

        TensorF t1 = layers(6, 5, 3, SEED + 8);
        TensorF t2 = layers(6, 5, 3, SEED + 8);
        assertSame(t1, t1.set(V, 1));
        t2.set(C, 1);
        assertBitsArray("set", t2.getArrayUnsafe(), t1.getArrayUnsafe());

        assertSame(t1, t1.append(V));
        t2.append(C);
        assertEquals("append depth", t2.numDepth(), t1.numDepth());
        assertBitsArray("append", t2.getArrayUnsafe(), t1.getArrayUnsafe());

        assertBitsArray("parent", parent, P.getArrayUnsafe());
    }

    @Test
    public void testProductsReadAViewArgumentLikeItsCopy() {
        MatrixF P = random(40, 30, SEED + 9);
        float[] parent = P.getArrayUnsafe().clone();
        MatrixF A = random(6, 5, SEED + 10);
        float[] target = A.getArrayUnsafe().clone();
        Set<String> names = new HashSet<>();
        for (Prod c : products()) {
            assertEquals("duplicate " + c.name, true, names.add(c.name));
            MatrixF[] views = new MatrixF[c.shapes.length / 2];
            MatrixF[] copies = new MatrixF[views.length];
            for (int i = 0; i < views.length; ++i) {
                views[i] = productView(P, c, i);
                copies[i] = views[i].copy();
            }
            MatrixF got = c.call.apply(A, views);
            MatrixF want = c.call.apply(A, copies);
            assertClose(c.name, want, got);
            assertBitsArray(c.name + ": target", target, A.getArrayUnsafe());
            assertBitsArray(c.name + ": parent", parent, P.getArrayUnsafe());
        }
    }

    @Test
    public void testOperandOfAnOrdinaryMatrix() {
        MatrixF A = random(7, 4, SEED);
        ReadAccess.OperandF op = ReadAccess.operand(A, null);
        assertSame(A.getArrayUnsafe(), op.array);
        assertEquals(0, op.offset);
        assertEquals(7, op.ld);
        MatrixF R = random(1, 5, SEED);
        assertEquals(1, ReadAccess.operand(R, R.getArrayUnsafe()).ld);
        assertSame(R.getArrayUnsafe(), ReadAccess.operand(R, R.getArrayUnsafe()).array);
    }

    @Test
    public void testOperandOfAViewReadsTheParent() {
        MatrixF P = random(9, 8, SEED);
        MatrixF V = Matrices.view(P, 2, 3, 6, 5);
        ReadAccess.OperandF op = ReadAccess.operand(V, new float[1]);
        assertSame(P.getArrayUnsafe(), op.array);
        assertEquals(3 * 9 + 2, op.offset);
        assertEquals(9, op.ld);

        ReadAccess.OperandF inner = ReadAccess.operand(Matrices.view(V, 1, 1, 3, 2), null);
        assertSame(P.getArrayUnsafe(), inner.array);
        assertEquals(4 * 9 + 3, inner.offset);
        assertEquals(9, inner.ld);

        ReadAccess.OperandF onOut = ReadAccess.operand(V, P.getArrayUnsafe());
        assertNotSame(P.getArrayUnsafe(), onOut.array);
        assertEquals(0, onOut.offset);
        assertEquals(5, onOut.ld);
        assertBitsArray("copy", V.copy().getArrayUnsafe(), onOut.array);
    }

    @Test
    public void testViewArgumentProductsDoNotCopy() {
        MatrixF A = random(6, 5, SEED + 10);
        for (Prod c : products()) {
            CountingMatrixF P = counting(40, 30, SEED + 9);
            MatrixF[] views = new MatrixF[c.shapes.length / 2];
            for (int i = 0; i < views.length; ++i) {
                views[i] = productView(P, c, i);
            }
            MatrixF got = c.call.apply(A, views);
            // the addend of timesPlus and timesMinus is copied into the result
            int addend = c.name.startsWith("timesPlus") || c.name.startsWith("timesMinus") ? 1 : 0;
            assertEquals(c.name + ": copies", addend, P.copies);
            MatrixF[] copies = new MatrixF[views.length];
            for (int i = 0; i < views.length; ++i) {
                copies[i] = views[i].copy();
            }
            assertClose(c.name, c.call.apply(A, copies), got);
        }

        // large enough for the threaded kernels
        CountingMatrixF P = counting(400, 300, SEED + 21);
        MatrixF L = random(250, 180, SEED + 22);
        MatrixF B = Matrices.view(P, 70, 50, 249, 209);
        MatrixF Bt = Matrices.view(P, 70, 50, 319, 229);
        MatrixF Bb = Matrices.view(P, 20, 40, 199, 219);
        MatrixF[] got = { L.mult(B, Matrices.createF(250, 160)), L.transAmultAdd(Bt, filled(180, 180)),
                L.transBmult(Bb, Matrices.createF(250, 180)), L.times(B) };
        assertEquals("large: copies", 0, P.copies);
        MatrixF[] want = { L.mult(B.copy(), Matrices.createF(250, 160)),
                L.transAmultAdd(Bt.copy(), filled(180, 180)), L.transBmult(Bb.copy(), Matrices.createF(250, 180)),
                L.times(B.copy()) };
        for (int i = 0; i < want.length; ++i) {
            assertClose("large " + i, want[i], got[i]);
        }
    }

    @Test
    public void testViewArgumentOnTheOutputIsCopiedOnce() {
        MatrixF A = random(6, 5, SEED + 11);
        CountingMatrixF C = counting(6, 4, SEED + 12);
        MatrixF C2 = random(6, 4, SEED + 12);
        MatrixF before = Matrices.view(C2, 1, 0, 5, 3).copy();
        A.multAdd(0.5f, Matrices.view(C, 1, 0, 5, 3), C);
        assertEquals("copies", 1, C.copies);
        A.multAdd(0.5f, before, C2);
        assertClose("multAdd", C2, C);
    }

    @Test
    public void testViewProductsDoNotCopy() {
        for (Prod c : products()) {
            if (c.name.equals("timesMany")) {
                continue;
            }
            // the arguments on a second parent, or on the receiver's own parent
            for (boolean sameParent : new boolean[] { false, true }) {
                String what = c.name + (sameParent ? " (same parent)" : "");
                CountingMatrixF PA = counting(40, 30, SEED + 23);
                CountingMatrixF PB = sameParent ? PA : counting(40, 30, SEED + 24);
                MatrixF V = Matrices.view(PA, 7, 6, 12, 10);
                MatrixF[] views = new MatrixF[c.shapes.length / 2];
                for (int i = 0; i < views.length; ++i) {
                    views[i] = productView(PB, c, i);
                }
                MatrixF got = c.call.apply(V, views);
                int addend = c.name.startsWith("timesPlus") || c.name.startsWith("timesMinus") ? 1 : 0;
                assertEquals(what + ": copies", addend, PA.copies + (sameParent ? 0 : PB.copies));
                MatrixF[] copies = new MatrixF[views.length];
                for (int i = 0; i < views.length; ++i) {
                    copies[i] = views[i].copy();
                }
                assertClose(what, c.call.apply(V.copy(), copies), got);
            }
        }

        CountingMatrixF S = counting(12, 11, SEED + 25);
        MatrixF W = Matrices.view(S, 2, 3, 8, 7);
        MatrixF[] got = { W.timesTransposed(), W.transposedTimes() };
        assertEquals("timesTransposed and transposedTimes: copies", 0, S.copies);
        assertClose("timesTransposed()", W.copy().timesTransposed(), got[0]);
        assertClose("transposedTimes()", W.copy().transposedTimes(), got[1]);

        CountingMatrixF P = counting(400, 300, SEED + 26);
        MatrixF L = Matrices.view(P, 70, 50, 319, 229);
        MatrixF B = random(180, 160, SEED + 27);
        MatrixF[] large = { L.times(B), L.transAmultAdd(Matrices.view(P, 10, 5, 259, 144), filled(180, 140)),
                L.timesTransposed() };
        assertEquals("large: copies", 0, P.copies);
        MatrixF Lc = L.copy();
        assertClose("large times", Lc.times(B), large[0]);
        assertClose("large transAmultAdd",
                Lc.transAmultAdd(Matrices.view(P, 10, 5, 259, 144).copy(), filled(180, 140)), large[1]);
        assertClose("large timesTransposed", Lc.timesTransposed(), large[2]);
    }

    @Test
    public void testViewProductsIntoTheParent() {
        // receiver region within C, receiver shape, argument shape, output shape
        ParentOut[] cases = {
                new ParentOut("multAdd", 0, 1, 6, 4, 4, 5, 6, 5, (V, B, C) -> V.multAdd(-1.0f, B, C)),
                new ParentOut("mult", 0, 1, 6, 4, 4, 5, 6, 5, (V, B, C) -> V.mult(2.0f, B, C)),
                new ParentOut("transAmultAdd", 1, 0, 4, 5, 4, 6, 5, 6, (V, B, C) -> V.transAmultAdd(B, C)),
                new ParentOut("transAmult", 1, 0, 4, 5, 4, 6, 5, 6, (V, B, C) -> V.transAmult(B, C)),
                new ParentOut("transBmultAdd", 0, 1, 5, 3, 4, 3, 5, 4, (V, B, C) -> V.transBmultAdd(0.5f, B, C)),
                new ParentOut("transBmult", 0, 1, 5, 3, 4, 3, 5, 4, (V, B, C) -> V.transBmult(B, C)),
                new ParentOut("transABmultAdd", 1, 1, 4, 5, 6, 4, 5, 6, (V, B, C) -> V.transABmultAdd(B, C)),
                new ParentOut("transABmult", 1, 1, 4, 5, 6, 4, 5, 6, (V, B, C) -> V.transABmult(-3.0f, B, C)) };
        for (ParentOut c : cases) {
            MatrixF B = random(c.argRows, c.argCols, SEED + 28);
            CountingMatrixF C1 = counting(c.outRows, c.outCols, SEED + 29);
            MatrixF C2 = random(c.outRows, c.outCols, SEED + 29);
            MatrixF before = Matrices.view(C2, c.r0, c.c0, c.r0 + c.rows - 1, c.c0 + c.cols - 1).copy();
            MatrixF V = Matrices.view(C1, c.r0, c.c0, c.r0 + c.rows - 1, c.c0 + c.cols - 1);
            assertSame(c.name, C1, c.call.apply(V, B, C1));
            assertEquals(c.name + ": copies", 1, C1.copies);
            c.call.apply(before, B, C2);
            assertClose(c.name, C2, C1);
        }

        // the argument, not the receiver, is on the output
        MatrixF V = Matrices.view(random(9, 9, SEED + 30), 2, 2, 7, 5);
        CountingMatrixF C1 = counting(6, 5, SEED + 31);
        MatrixF C2 = random(6, 5, SEED + 31);
        MatrixF before = Matrices.view(C2, 1, 0, 4, 4).copy();
        V.multAdd(Matrices.view(C1, 1, 0, 4, 4), C1);
        assertEquals("argument on the output: copies", 1, C1.copies);
        V.copy().multAdd(before, C2);
        assertClose("argument on the output", C2, C1);
        CountingMatrixF D1 = counting(6, 5, SEED + 31);
        MatrixF D2 = random(6, 5, SEED + 31);
        MatrixF beforeD = Matrices.view(D2, 1, 0, 4, 4).copy();
        V.mult(Matrices.view(D1, 1, 0, 4, 4), D1);
        assertEquals("argument on the zeroed output: copies", 1, D1.copies);
        V.copy().mult(beforeD, D2);
        assertClose("argument on the zeroed output", D2, D1);
    }

    @Test
    public void testTimesManyStillCopiesOnce() {
        CountingMatrixF P = counting(10, 9, SEED + 32);
        MatrixF V = Matrices.view(P, 2, 1, 7, 5);
        MatrixF B = random(5, 4, SEED + 33);
        MatrixF C = random(4, 3, SEED + 34);
        MatrixF D = random(3, 2, SEED + 35);
        MatrixF got = V.timesMany(B, C, D);
        assertEquals("copies", 1, P.copies);
        assertClose("timesMany", V.copy().timesMany(B, C, D), got);
    }

    static Prod[] products() {
        return new Prod[] {
                new Prod("mult", s(5, 4), (T, x) -> T.mult(x[0], Matrices.createF(6, 4))),
                new Prod("mult alpha", s(5, 4), (T, x) -> T.mult(-0.5f, x[0], Matrices.createF(6, 4))),
                new Prod("multAdd", s(5, 4), (T, x) -> T.multAdd(x[0], filled(6, 4))),
                new Prod("multAdd alpha", s(5, 4), (T, x) -> T.multAdd(1.5f, x[0], filled(6, 4))),
                new Prod("transABmult", s(4, 6), (T, x) -> T.transABmult(x[0], Matrices.createF(5, 4))),
                new Prod("transABmult alpha", s(4, 6), (T, x) -> T.transABmult(2.0f, x[0], Matrices.createF(5, 4))),
                new Prod("transABmultAdd", s(4, 6), (T, x) -> T.transABmultAdd(x[0], filled(5, 4))),
                new Prod("transABmultAdd alpha", s(4, 6), (T, x) -> T.transABmultAdd(-1.0f, x[0], filled(5, 4))),
                new Prod("transAmult", s(6, 4), (T, x) -> T.transAmult(x[0], Matrices.createF(5, 4))),
                new Prod("transAmult alpha", s(6, 4), (T, x) -> T.transAmult(3.0f, x[0], Matrices.createF(5, 4))),
                new Prod("transAmultAdd", s(6, 4), (T, x) -> T.transAmultAdd(x[0], filled(5, 4))),
                new Prod("transAmultAdd alpha", s(6, 4), (T, x) -> T.transAmultAdd(0.25f, x[0], filled(5, 4))),
                new Prod("transBmult", s(4, 5), (T, x) -> T.transBmult(x[0], Matrices.createF(6, 4))),
                new Prod("transBmult alpha", s(4, 5), (T, x) -> T.transBmult(-2.0f, x[0], Matrices.createF(6, 4))),
                new Prod("transBmultAdd", s(4, 5), (T, x) -> T.transBmultAdd(x[0], filled(6, 4))),
                new Prod("transBmultAdd alpha", s(4, 5), (T, x) -> T.transBmultAdd(0.5f, x[0], filled(6, 4))),
                new Prod("times", s(5, 4), (T, x) -> T.times(x[0])),
                new Prod("timesTimes", s(5, 4, 4, 3), (T, x) -> T.timesTimes(x[0], x[1])),
                new Prod("timesMany", s(5, 4, 4, 3, 3, 2), (T, x) -> T.timesMany(x[0], x[1], x[2])),
                new Prod("timesTransposed", s(3, 5), (T, x) -> T.timesTransposed(x[0])),
                new Prod("transposedTimes", s(6, 3), (T, x) -> T.transposedTimes(x[0])),
                new Prod("timesPlus", s(5, 4, 6, 4), (T, x) -> T.timesPlus(x[0], x[1])),
                new Prod("timesMinus", s(5, 4, 6, 4), (T, x) -> T.timesMinus(x[0], x[1])) };
    }

    // the i-th argument of a product case, at an offset that differs per argument
    static MatrixF productView(MatrixF P, Prod c, int i) {
        int r0 = 1 + 3 * i;
        int c0 = 2 + 2 * i;
        return Matrices.view(P, r0, c0, r0 + c.shapes[2 * i] - 1, c0 + c.shapes[2 * i + 1] - 1);
    }

    @Test
    public void testProductWithTheOutputBeingTheParentOfTheArgument() {
        MatrixF A = random(6, 5, SEED + 11);
        MatrixF C1 = random(6, 4, SEED + 12);
        MatrixF C2 = random(6, 4, SEED + 12);
        MatrixF before = Matrices.view(C2, 1, 0, 5, 3).copy();
        assertSame(C1, A.multAdd(-1.0f, Matrices.view(C1, 1, 0, 5, 3), C1));
        A.multAdd(-1.0f, before, C2);
        assertClose("multAdd", C2, C1);

        MatrixF At = random(4, 5, SEED + 13);
        MatrixF D1 = random(5, 4, SEED + 14);
        MatrixF D2 = random(5, 4, SEED + 14);
        MatrixF beforeT = Matrices.view(D2, 1, 0, 4, 3).copy();
        assertSame(D1, At.transAmultAdd(2.0f, Matrices.view(D1, 1, 0, 4, 3), D1));
        At.transAmultAdd(2.0f, beforeT, D2);
        assertClose("transAmultAdd", D2, D1);
    }

    @Test
    public void testZeroedOutputIsNotReadThroughAView() {
        // A shape, C shape, B region within C
        OutputAlias[] cases = {
                new OutputAlias("mult", 6, 5, 6, 4, 1, 0, 5, 3, (A, B, C) -> A.mult(B, C)),
                new OutputAlias("mult alpha", 6, 5, 6, 4, 1, 0, 5, 3, (A, B, C) -> A.mult(-2.0f, B, C)),
                new OutputAlias("transAmult", 4, 5, 5, 4, 1, 0, 4, 3, (A, B, C) -> A.transAmult(B, C)),
                new OutputAlias("transAmult alpha", 4, 5, 5, 4, 1, 0, 4, 3, (A, B, C) -> A.transAmult(0.5f, B, C)),
                new OutputAlias("transBmult", 7, 5, 7, 6, 1, 0, 6, 4, (A, B, C) -> A.transBmult(B, C)),
                new OutputAlias("transBmult alpha", 7, 5, 7, 6, 1, 0, 6, 4, (A, B, C) -> A.transBmult(3.0f, B, C)),
                new OutputAlias("transABmult", 4, 6, 6, 5, 1, 1, 5, 4, (A, B, C) -> A.transABmult(B, C)),
                new OutputAlias("transABmult alpha", 4, 6, 6, 5, 1, 1, 5, 4,
                        (A, B, C) -> A.transABmult(-1.0f, B, C)) };
        for (OutputAlias c : cases) {
            MatrixF A = random(c.rows, c.cols, SEED + 19);
            MatrixF C1 = random(c.outRows, c.outCols, SEED + 20);
            MatrixF C2 = random(c.outRows, c.outCols, SEED + 20);
            MatrixF before = Matrices.view(C2, c.r0, c.c0, c.r1, c.c1).copy();
            assertSame(c.name, C1, c.call.apply(A, Matrices.view(C1, c.r0, c.c0, c.r1, c.c1), C1));
            c.call.apply(A, before, C2);
            assertClose(c.name, C2, C1);
        }
    }

    @Test
    public void testMomentsViewIsRefusedBeforeAIsWritten() {
        MatrixF Pr = random(3, 9, SEED + 15);
        MatrixF Pc = random(12, 4, SEED + 16);
        float[] rowParent = Pr.getArrayUnsafe().clone();
        float[] colParent = Pc.getArrayUnsafe().clone();
        MatrixF A = random(8, 5, SEED + 17);
        float[] a = A.getArrayUnsafe().clone();
        Statistics.MomentsF[] columns = {
                new Statistics.MomentsF(Matrices.view(Pr, 1, 2, 1, 6), Matrices.createF(1, 5)),
                new Statistics.MomentsF(Matrices.createF(1, 5), Matrices.view(Pr, 2, 0, 2, 4)),
                new Statistics.MomentsF(Matrices.view(Pr, 0, 0, 1, 2), null) };
        Statistics.MomentsF[] rows = {
                new Statistics.MomentsF(Matrices.view(Pc, 2, 1, 9, 1), Matrices.createF(8, 1)),
                new Statistics.MomentsF(Matrices.createF(8, 1), Matrices.view(Pc, 0, 3, 7, 3)),
                new Statistics.MomentsF(null, Matrices.view(Pc, 0, 0, 2, 2)) };
        for (int i = 0; i < columns.length; ++i) {
            Statistics.MomentsF m = columns[i];
            assertRefused("zscoreColumnsInplace " + i, m, () -> Statistics.zscoreColumnsInplace(A, m));
        }
        for (int i = 0; i < rows.length; ++i) {
            Statistics.MomentsF m = rows[i];
            assertRefused("zscoreRowsInplace " + i, m, () -> Statistics.zscoreRowsInplace(A, m));
        }
        assertBitsArray("A", a, A.getArrayUnsafe());
        assertBitsArray("row parent", rowParent, Pr.getArrayUnsafe());
        assertBitsArray("column parent", colParent, Pc.getArrayUnsafe());
    }

    @Test
    public void testEveryMatrixFParameterIsClassified() {
        // implementations must not add public MatrixF parameters of their own
        for (Class<?> impl : new Class<?>[] { MatrixFBase.class, SimpleMatrixF.class }) {
            for (Method m : impl.getMethods()) {
                if (hasMatrixFParameter(m)) {
                    try {
                        MatrixF.class.getMethod(m.getName(), m.getParameterTypes());
                    } catch (NoSuchMethodException e) {
                        fail(impl.getSimpleName() + " declares " + key(impl.getSimpleName(), m, -1));
                    }
                }
            }
        }

        Set<String> found = new TreeSet<>();
        collect(found, "MatrixF", Arrays.asList(MatrixF.class.getMethods()));
        collect(found, "ComplexMatrixF", Arrays.asList(ComplexMatrixF.class.getMethods()));
        for (Class<?> c : new Class<?>[] { Matrices.class, Statistics.class, TensorF.class }) {
            List<Executable> members = new ArrayList<>(Arrays.asList(c.getDeclaredMethods()));
            members.addAll(Arrays.asList(c.getConstructors()));
            collect(found, c.getSimpleName(), members);
        }

        Param[] table = params();
        Set<String> listed = new TreeSet<>();
        for (Param p : table) {
            assertEquals("duplicate " + p.key, true, listed.add(p.key));
        }
        Set<String> missing = new TreeSet<>(found);
        missing.removeAll(listed);
        Set<String> unknown = new TreeSet<>(listed);
        unknown.removeAll(found);
        assertEquals("not classified", "[]", missing.toString());
        assertEquals("not a parameter", "[]", unknown.toString());

        MatrixF P = random(60, 40, SEED + 18).addInplace(30.0f, Matrices.identityF(60).selectConsecutiveColumns(0, 39));
        float[] parent = P.getArrayUnsafe().clone();
        for (Param p : table) {
            MatrixF V = Matrices.view(P, 4, 4, 4 + p.rows - 1, 4 + p.cols - 1);
            if (p.kind == Kind.WRITE) {
                Ctx ctx = new Ctx(V);
                try {
                    p.call.apply(ctx);
                    fail(p.key + ": no exception");
                } catch (UnsupportedOperationException expected) {
                    ctx.assertUntouched(p.key);
                }
            } else {
                Object got = p.call.apply(new Ctx(V));
                Object want = p.call.apply(new Ctx(V.copy()));
                if (p.kind == Kind.BITS) {
                    assertResult(p.key, want, got);
                } else if (p.kind == Kind.CLOSE) {
                    if (want instanceof ComplexMatrixF) {
                        assertClose(p.key, (ComplexMatrixF) want, (ComplexMatrixF) got);
                    } else {
                        assertClose(p.key, (MatrixF) want, (MatrixF) got);
                    }
                } else {
                    float[] w = ((MatrixF) want).getArrayUnsafe().clone();
                    float[] g = ((MatrixF) got).getArrayUnsafe().clone();
                    Arrays.sort(w);
                    Arrays.sort(g);
                    assertBitsArray(p.key, w, g);
                }
            }
            assertBitsArray(p.key + ": parent", parent, P.getArrayUnsafe());
        }
    }

    // every (member, MatrixF parameter) pair; x is the view or its copy
    static Param[] params() {
        String B = "(MatrixF,MatrixF)#";
        String aB = "(float,MatrixF,MatrixF)#";
        return new Param[] {
                // MatrixF: products read B and write C
                close("MatrixF.mult" + B + 0, 5, 4, c -> c.t(6, 5).mult(c.x, c.t(6, 4))),
                close("MatrixF.mult" + aB + 1, 5, 4, c -> c.t(6, 5).mult(0.5f, c.x, c.t(6, 4))),
                close("MatrixF.multAdd" + B + 0, 5, 4, c -> c.t(6, 5).multAdd(c.x, c.t(6, 4))),
                close("MatrixF.multAdd" + aB + 1, 5, 4, c -> c.t(6, 5).multAdd(2.0f, c.x, c.t(6, 4))),
                close("MatrixF.transAmult" + B + 0, 6, 4, c -> c.t(6, 5).transAmult(c.x, c.t(5, 4))),
                close("MatrixF.transAmult" + aB + 1, 6, 4, c -> c.t(6, 5).transAmult(0.5f, c.x, c.t(5, 4))),
                close("MatrixF.transAmultAdd" + B + 0, 6, 4, c -> c.t(6, 5).transAmultAdd(c.x, c.t(5, 4))),
                close("MatrixF.transAmultAdd" + aB + 1, 6, 4,
                        c -> c.t(6, 5).transAmultAdd(2.0f, c.x, c.t(5, 4))),
                close("MatrixF.transBmult" + B + 0, 4, 5, c -> c.t(6, 5).transBmult(c.x, c.t(6, 4))),
                close("MatrixF.transBmult" + aB + 1, 4, 5, c -> c.t(6, 5).transBmult(0.5f, c.x, c.t(6, 4))),
                close("MatrixF.transBmultAdd" + B + 0, 4, 5, c -> c.t(6, 5).transBmultAdd(c.x, c.t(6, 4))),
                close("MatrixF.transBmultAdd" + aB + 1, 4, 5,
                        c -> c.t(6, 5).transBmultAdd(2.0f, c.x, c.t(6, 4))),
                close("MatrixF.transABmult" + B + 0, 4, 6, c -> c.t(6, 5).transABmult(c.x, c.t(5, 4))),
                close("MatrixF.transABmult" + aB + 1, 4, 6, c -> c.t(6, 5).transABmult(0.5f, c.x, c.t(5, 4))),
                close("MatrixF.transABmultAdd" + B + 0, 4, 6, c -> c.t(6, 5).transABmultAdd(c.x, c.t(5, 4))),
                close("MatrixF.transABmultAdd" + aB + 1, 4, 6,
                        c -> c.t(6, 5).transABmultAdd(2.0f, c.x, c.t(5, 4))),
                write("MatrixF.mult" + B + 1, 6, 4, c -> c.t(6, 5).mult(c.t(5, 4), c.x)),
                write("MatrixF.mult" + aB + 2, 6, 4, c -> c.t(6, 5).mult(0.5f, c.t(5, 4), c.x)),
                write("MatrixF.multAdd" + B + 1, 6, 4, c -> c.t(6, 5).multAdd(c.t(5, 4), c.x)),
                write("MatrixF.multAdd" + aB + 2, 6, 4, c -> c.t(6, 5).multAdd(2.0f, c.t(5, 4), c.x)),
                write("MatrixF.transAmult" + B + 1, 5, 4, c -> c.t(6, 5).transAmult(c.t(6, 4), c.x)),
                write("MatrixF.transAmult" + aB + 2, 5, 4, c -> c.t(6, 5).transAmult(0.5f, c.t(6, 4), c.x)),
                write("MatrixF.transAmultAdd" + B + 1, 5, 4, c -> c.t(6, 5).transAmultAdd(c.t(6, 4), c.x)),
                write("MatrixF.transAmultAdd" + aB + 2, 5, 4,
                        c -> c.t(6, 5).transAmultAdd(2.0f, c.t(6, 4), c.x)),
                write("MatrixF.transBmult" + B + 1, 6, 4, c -> c.t(6, 5).transBmult(c.t(4, 5), c.x)),
                write("MatrixF.transBmult" + aB + 2, 6, 4, c -> c.t(6, 5).transBmult(0.5f, c.t(4, 5), c.x)),
                write("MatrixF.transBmultAdd" + B + 1, 6, 4, c -> c.t(6, 5).transBmultAdd(c.t(4, 5), c.x)),
                write("MatrixF.transBmultAdd" + aB + 2, 6, 4,
                        c -> c.t(6, 5).transBmultAdd(2.0f, c.t(4, 5), c.x)),
                write("MatrixF.transABmult" + B + 1, 5, 4, c -> c.t(6, 5).transABmult(c.t(4, 6), c.x)),
                write("MatrixF.transABmult" + aB + 2, 5, 4, c -> c.t(6, 5).transABmult(0.5f, c.t(4, 6), c.x)),
                write("MatrixF.transABmultAdd" + B + 1, 5, 4, c -> c.t(6, 5).transABmultAdd(c.t(4, 6), c.x)),
                write("MatrixF.transABmultAdd" + aB + 2, 5, 4,
                        c -> c.t(6, 5).transABmultAdd(2.0f, c.t(4, 6), c.x)),
                // MatrixF: element-wise reads, in place on the receiver
                bits("MatrixF.add" + B + 0, 6, 5, c -> c.t(6, 5).add(c.x, c.t(6, 5))),
                bits("MatrixF.add" + aB + 1, 6, 5, c -> c.t(6, 5).add(-1.5f, c.x, c.t(6, 5))),
                write("MatrixF.add" + B + 1, 6, 5, c -> c.t(6, 5).add(c.t(6, 5), c.x)),
                write("MatrixF.add" + aB + 2, 6, 5, c -> c.t(6, 5).add(-1.5f, c.t(6, 5), c.x)),
                bits("MatrixF.addInplace(MatrixF)#0", 6, 5, c -> c.t(6, 5).addInplace(c.x)),
                bits("MatrixF.addInplace(float,MatrixF)#1", 6, 5, c -> c.t(6, 5).addInplace(0.5f, c.x)),
                bits("MatrixF.addBroadcastedVectorInplace(MatrixF)#0", 6, 1,
                        c -> c.t(6, 5).addBroadcastedVectorInplace(c.x)),
                bits("MatrixF.mulBroadcastedVectorInplace(MatrixF)#0", 6, 1,
                        c -> c.t(6, 5).mulBroadcastedVectorInplace(c.x)),
                bits("MatrixF.divBroadcastedVectorInplace(MatrixF)#0", 6, 1,
                        c -> c.t(6, 5).divBroadcastedVectorInplace(c.x)),
                bits("MatrixF.addBroadcastedRowVectorInplace(MatrixF)#0", 1, 5,
                        c -> c.t(6, 5).addBroadcastedRowVectorInplace(c.x)),
                bits("MatrixF.mulBroadcastedRowVectorInplace(MatrixF)#0", 1, 5,
                        c -> c.t(6, 5).mulBroadcastedRowVectorInplace(c.x)),
                bits("MatrixF.divBroadcastedRowVectorInplace(MatrixF)#0", 1, 5,
                        c -> c.t(6, 5).divBroadcastedRowVectorInplace(c.x)),
                bits("MatrixF.setSubmatrixInplace(int,int,MatrixF,int,int,int,int)#2", 10, 4,
                        c -> c.t(12, 6).setSubmatrixInplace(1, 1, c.x, 0, 0, 9, 3)),
                bits("MatrixF.setInplaceUpperTrapezoidal(MatrixF)#0", 7, 6,
                        c -> c.t(6, 5).setInplaceUpperTrapezoidal(c.x)),
                bits("MatrixF.setInplaceLowerTrapezoidal(MatrixF)#0", 7, 6,
                        c -> c.t(6, 5).setInplaceLowerTrapezoidal(c.x)),
                bits("MatrixF.setColumnInplace(int,MatrixF)#1", 6, 1, c -> c.t(6, 5).setColumnInplace(2, c.x)),
                bits("MatrixF.setInplace(MatrixF)#0", 6, 5, c -> c.t(6, 5).setInplace(c.x)),
                bits("MatrixF.setInplace(float,MatrixF)#1", 6, 5, c -> c.t(6, 5).setInplace(3.0f, c.x)),
                bits("MatrixF.hadamard" + B + 0, 6, 5, c -> c.t(6, 5).hadamard(c.x, c.t(6, 5))),
                write("MatrixF.hadamard" + B + 1, 6, 5, c -> c.t(6, 5).hadamard(c.t(6, 5), c.x)),
                close("MatrixF.solve" + B + 0, 5, 2, c -> c.sq(5).solve(c.x, c.t(5, 2))),
                write("MatrixF.solve" + B + 1, 5, 2, c -> c.sq(5).solve(c.t(5, 2), c.x)),
                write("MatrixF.trans(MatrixF)#0", 5, 6, c -> c.t(6, 5).trans(c.x)),
                write("MatrixF.scale(float,MatrixF)#1", 6, 5, c -> c.t(6, 5).scale(2.0f, c.x)),
                write("MatrixF.inv(MatrixF)#0", 5, 5, c -> c.sq(5).inv(c.x)),
                write("MatrixF.submatrix(int,int,int,int,MatrixF,int,int)#4", 3, 3,
                        c -> c.t(6, 5).submatrix(0, 0, 2, 2, c.x, 0, 0)),
                // MatrixF: reads that allocate their result
                bits("MatrixF.plusBroadcastedVector(MatrixF)#0", 6, 1, c -> c.t(6, 5).plusBroadcastedVector(c.x)),
                bits("MatrixF.mulBroadcastedVector(MatrixF)#0", 6, 1, c -> c.t(6, 5).mulBroadcastedVector(c.x)),
                bits("MatrixF.divBroadcastedVector(MatrixF)#0", 6, 1, c -> c.t(6, 5).divBroadcastedVector(c.x)),
                bits("MatrixF.plusBroadcastedRowVector(MatrixF)#0", 1, 5,
                        c -> c.t(6, 5).plusBroadcastedRowVector(c.x)),
                bits("MatrixF.mulBroadcastedRowVector(MatrixF)#0", 1, 5,
                        c -> c.t(6, 5).mulBroadcastedRowVector(c.x)),
                bits("MatrixF.divBroadcastedRowVector(MatrixF)#0", 1, 5,
                        c -> c.t(6, 5).divBroadcastedRowVector(c.x)),
                bits("MatrixF.hadamardTransposed(MatrixF)#0", 5, 6, c -> c.t(6, 5).hadamardTransposed(c.x)),
                bits("MatrixF.transposedHadamard(MatrixF)#0", 5, 6, c -> c.t(6, 5).transposedHadamard(c.x)),
                bits("MatrixF.plus(MatrixF)#0", 6, 5, c -> c.t(6, 5).plus(c.x)),
                bits("MatrixF.minus(MatrixF)#0", 6, 5, c -> c.t(6, 5).minus(c.x)),
                bits("MatrixF.hadamard(MatrixF)#0", 6, 5, c -> c.t(6, 5).hadamard(c.x)),
                bits("MatrixF.appendColumn(MatrixF)#0", 6, 1, c -> c.t(6, 5).appendColumn(c.x)),
                bits("MatrixF.appendMatrix(MatrixF)#0", 6, 3, c -> c.t(6, 5).appendMatrix(c.x)),
                close("MatrixF.times(MatrixF)#0", 5, 4, c -> c.t(6, 5).times(c.x)),
                close("MatrixF.timesTimes" + B + 0, 5, 4, c -> c.t(6, 5).timesTimes(c.x, c.t(4, 3))),
                close("MatrixF.timesTimes" + B + 1, 4, 3, c -> c.t(6, 5).timesTimes(c.t(5, 4), c.x)),
                close("MatrixF.timesMany(MatrixF,MatrixF[])#0", 5, 4,
                        c -> c.t(6, 5).timesMany(c.x, c.t(4, 3), c.t(3, 2))),
                close("MatrixF.timesMany(MatrixF,MatrixF[])#1", 4, 3,
                        c -> c.t(6, 5).timesMany(c.t(5, 4), c.x, c.t(3, 2))),
                close("MatrixF.timesPlus" + B + 0, 5, 4, c -> c.t(6, 5).timesPlus(c.x, c.t(6, 4))),
                close("MatrixF.timesPlus" + B + 1, 6, 4, c -> c.t(6, 5).timesPlus(c.t(5, 4), c.x)),
                close("MatrixF.timesMinus" + B + 0, 5, 4, c -> c.t(6, 5).timesMinus(c.x, c.t(6, 4))),
                close("MatrixF.timesMinus" + B + 1, 6, 4, c -> c.t(6, 5).timesMinus(c.t(5, 4), c.x)),
                close("MatrixF.timesTransposed(MatrixF)#0", 3, 5, c -> c.t(6, 5).timesTransposed(c.x)),
                close("MatrixF.transposedTimes(MatrixF)#0", 6, 3, c -> c.t(6, 5).transposedTimes(c.x)),
                close("MatrixF.mldivide(MatrixF)#0", 5, 2, c -> c.sq(5).mldivide(c.x)),
                close("MatrixF.mrdivide(MatrixF)#0", 5, 5, c -> c.t(4, 5).mrdivide(c.x)),
                close("ComplexMatrixF.times(MatrixF)#0", 6, 5,
                        c -> Matrices.randomUniformComplexF(4, 6, SEED).times(c.x)),
                // Matrices
                bits("Matrices.view(MatrixF,int,int,int,int)#0", 6, 5, c -> Matrices.view(c.x, 1, 1, 4, 3).copy()),
                bits("Matrices.embed(int,int,MatrixF)#2", 6, 5, c -> Matrices.embed(8, 7, c.x)),
                bits("Matrices.sameDimF(MatrixF)#0", 6, 5, c -> Matrices.sameDimF(c.x)),
                bits("Matrices.serializeF(MatrixF,Path)#0", 6, 5, c -> serializeToFile(c.x)),
                bits("Matrices.serializeF(MatrixF,OutputStream)#0", 6, 5, c -> serialize(c.x)),
                bits("Matrices.convert(MatrixF)#0", 6, 5, c -> Matrices.convert(c.x)),
                bits("Matrices.convertToComplex(MatrixF)#0", 6, 5, c -> Matrices.convertToComplex(c.x)),
                bits("Matrices.round(MatrixF,int)#0", 6, 5, c -> Matrices.round(c.x, 4)),
                bits("Matrices.timeDelayEmbeddingF(MatrixF,int)#0", 6, 5, c -> Matrices.timeDelayEmbeddingF(c.x, 2)),
                bits("Matrices.sumRows(MatrixF)#0", 6, 5, c -> Matrices.sumRows(c.x)),
                bits("Matrices.sumColumns(MatrixF)#0", 6, 5, c -> Matrices.sumColumns(c.x)),
                bits("Matrices.rowsAverage(MatrixF)#0", 6, 5, c -> Matrices.rowsAverage(c.x)),
                bits("Matrices.colsAverage(MatrixF)#0", 6, 5, c -> Matrices.colsAverage(c.x)),
                bits("Matrices.distance(MatrixF,MatrixF)#0", 6, 5, c -> Matrices.distance(c.x, c.t(6, 5))),
                bits("Matrices.distance(MatrixF,MatrixF)#1", 6, 5, c -> Matrices.distance(c.t(6, 5), c.x)),
                bits("Matrices.approxEqual(MatrixF,MatrixF)#0", 6, 5, c -> Matrices.approxEqual(c.x, c.x.copy())),
                bits("Matrices.approxEqual(MatrixF,MatrixF)#1", 6, 5, c -> Matrices.approxEqual(c.t(6, 5), c.x)),
                bits("Matrices.approxEqual(MatrixF,MatrixF,float)#0", 6, 5,
                        c -> Matrices.approxEqual(c.x, c.x.copy(), 1.0e-3f)),
                bits("Matrices.approxEqual(MatrixF,MatrixF,float)#1", 6, 5,
                        c -> Matrices.approxEqual(c.t(6, 5), c.x, 1.0f)),
                bits("Matrices.approxEqual(MatrixF,MatrixF,float,float)#0", 6, 5,
                        c -> Matrices.approxEqual(c.x, c.t(6, 5), 0.0f, 1.0f)),
                bits("Matrices.approxEqual(MatrixF,MatrixF,float,float)#1", 6, 5,
                        c -> Matrices.approxEqual(c.t(6, 5), c.x, 0.0f, 0.1f)),
                bits("Matrices.numericalRank(MatrixF)#0", 6, 5, c -> Matrices.numericalRank(c.x)),
                bits("Matrices.numericalRank(MatrixF,float)#0", 6, 5, c -> Matrices.numericalRank(c.x, 1.0f)),
                // Statistics
                bits("Statistics.centerColumns(MatrixF)#0", 6, 5, c -> Statistics.centerColumns(c.x)),
                bits("Statistics.zscoreColumns(MatrixF)#0", 6, 5, c -> Statistics.zscoreColumns(c.x)),
                bits("Statistics.zscoreRows(MatrixF)#0", 6, 5, c -> Statistics.zscoreRows(c.x)),
                bits("Statistics.rescale(MatrixF,float,float)#0", 6, 5, c -> Statistics.rescale(c.x, -1.0f, 1.0f)),
                bits("Statistics.shuffleColumns(MatrixF,long)#0", 6, 5, c -> Statistics.shuffleColumns(c.x, 7L)),
                bits("Statistics.shuffleRows(MatrixF,long)#0", 6, 5, c -> Statistics.shuffleRows(c.x, 7L)),
                permutation("Statistics.shuffleColumns(MatrixF)#0", 6, 5, c -> Statistics.shuffleColumns(c.x)),
                permutation("Statistics.shuffleRows(MatrixF)#0", 6, 5, c -> Statistics.shuffleRows(c.x)),
                write("Statistics.centerColumnsInplace(MatrixF)#0", 6, 5, c -> Statistics.centerColumnsInplace(c.x)),
                write("Statistics.zscoreColumnsInplace(MatrixF)#0", 6, 5, c -> Statistics.zscoreColumnsInplace(c.x)),
                write("Statistics.zscoreRowsInplace(MatrixF)#0", 6, 5, c -> Statistics.zscoreRowsInplace(c.x)),
                write("Statistics.zscoreColumnsInplace(MatrixF,MomentsF)#0", 6, 5,
                        c -> Statistics.zscoreColumnsInplace(c.x, c.moments())),
                write("Statistics.zscoreRowsInplace(MatrixF,MomentsF)#0", 6, 5,
                        c -> Statistics.zscoreRowsInplace(c.x, c.moments())),
                write("Statistics.zscoreColumnsInplace(MatrixF,MomentsF)#1", 1, 5,
                        c -> Statistics.zscoreColumnsInplace(c.t(8, 5), new Statistics.MomentsF(c.x, null))),
                write("Statistics.zscoreRowsInplace(MatrixF,MomentsF)#1", 8, 1,
                        c -> Statistics.zscoreRowsInplace(c.t(8, 5), new Statistics.MomentsF(null, c.x))),
                write("Statistics.rescaleInplace(MatrixF,float,float)#0", 6, 5,
                        c -> Statistics.rescaleInplace(c.x, 0.0f, 1.0f)),
                write("Statistics.shuffleColumnsInplace(MatrixF)#0", 6, 5, c -> Statistics.shuffleColumnsInplace(c.x)),
                write("Statistics.shuffleColumnsInplace(MatrixF,long)#0", 6, 5,
                        c -> Statistics.shuffleColumnsInplace(c.x, 7L)),
                write("Statistics.shuffleRowsInplace(MatrixF)#0", 6, 5, c -> Statistics.shuffleRowsInplace(c.x)),
                write("Statistics.shuffleRowsInplace(MatrixF,long)#0", 6, 5,
                        c -> Statistics.shuffleRowsInplace(c.x, 7L)),
                // TensorF
                bits("TensorF.<init>(MatrixF)#0", 6, 5, c -> new TensorF(c.x)),
                bits("TensorF.set(MatrixF,int)#0", 6, 5, c -> layers(6, 5, 3, SEED).set(c.x, 1)),
                bits("TensorF.append(MatrixF)#0", 6, 5, c -> layers(6, 5, 3, SEED).append(c.x)) };
    }

    // ---------------------------------------------------------------- cases

    static final class ArgCase {
        final String name;
        final int rows;
        final int cols;
        final int argRows;
        final int argCols;
        final BiFunction<MatrixF, MatrixF, Object> call;

        ArgCase(String name, int rows, int cols, int argRows, int argCols, BiFunction<MatrixF, MatrixF, Object> call) {
            this.name = name;
            this.rows = rows;
            this.cols = cols;
            this.argRows = argRows;
            this.argCols = argCols;
            this.call = call;
        }
    }

    static final class Overlap {
        final String name;
        final int rows;
        final int cols;
        final int r0;
        final int c0;
        final int r1;
        final int c1;
        final BiConsumer<MatrixF, MatrixF> call;

        Overlap(String name, int rows, int cols, int r0, int c0, int r1, int c1, BiConsumer<MatrixF, MatrixF> call) {
            this.name = name;
            this.rows = rows;
            this.cols = cols;
            this.r0 = r0;
            this.c0 = c0;
            this.r1 = r1;
            this.c1 = c1;
            this.call = call;
        }
    }

    static final class Stat {
        final String name;
        final boolean exact;
        final int[] a;
        final int[] b;
        final BiFunction<MatrixF, MatrixF, Object> call;

        Stat(String name, boolean exact, int[] a, int[] b, BiFunction<MatrixF, MatrixF, Object> call) {
            this.name = name;
            this.exact = exact;
            this.a = a;
            this.b = b;
            this.call = call;
        }
    }

    // counts the block copies a view of this matrix makes
    static final class CountingMatrixF extends SimpleMatrixF {
        int copies;

        CountingMatrixF(int rows, int cols) {
            super(rows, cols);
        }

        @Override
        public MatrixF submatrix(int r0, int c0, int r1, int c1, MatrixF B, int rb, int cb) {
            ++copies;
            return super.submatrix(r0, c0, r1, c1, B, rb, cb);
        }
    }

    static CountingMatrixF counting(int rows, int cols, long seed) {
        CountingMatrixF m = new CountingMatrixF(rows, cols);
        m.setInplace(random(rows, cols, seed));
        return m;
    }

    interface TriFunction {
        MatrixF apply(MatrixF A, MatrixF B, MatrixF C);
    }

    static final class ParentOut {
        final String name;
        final int r0;
        final int c0;
        final int rows;
        final int cols;
        final int argRows;
        final int argCols;
        final int outRows;
        final int outCols;
        final TriFunction call;

        ParentOut(String name, int r0, int c0, int rows, int cols, int argRows, int argCols, int outRows,
                int outCols, TriFunction call) {
            this.name = name;
            this.r0 = r0;
            this.c0 = c0;
            this.rows = rows;
            this.cols = cols;
            this.argRows = argRows;
            this.argCols = argCols;
            this.outRows = outRows;
            this.outCols = outCols;
            this.call = call;
        }
    }

    static final class OutputAlias {
        final String name;
        final int rows;
        final int cols;
        final int outRows;
        final int outCols;
        final int r0;
        final int c0;
        final int r1;
        final int c1;
        final TriFunction call;

        OutputAlias(String name, int rows, int cols, int outRows, int outCols, int r0, int c0, int r1, int c1,
                TriFunction call) {
            this.name = name;
            this.rows = rows;
            this.cols = cols;
            this.outRows = outRows;
            this.outCols = outCols;
            this.r0 = r0;
            this.c0 = c0;
            this.r1 = r1;
            this.c1 = c1;
            this.call = call;
        }
    }

    static final class Prod {
        final String name;
        final int[] shapes;
        final BiFunction<MatrixF, MatrixF[], MatrixF> call;

        Prod(String name, int[] shapes, BiFunction<MatrixF, MatrixF[], MatrixF> call) {
            this.name = name;
            this.shapes = shapes;
            this.call = call;
        }
    }

    enum Kind {
        BITS, CLOSE, PERMUTATION, WRITE
    }

    static final class Param {
        final String key;
        final Kind kind;
        final int rows;
        final int cols;
        final Function<Ctx, Object> call;

        Param(String key, Kind kind, int rows, int cols, Function<Ctx, Object> call) {
            this.key = key;
            this.kind = kind;
            this.rows = rows;
            this.cols = cols;
            this.call = call;
        }
    }

    static Param bits(String key, int rows, int cols, Function<Ctx, Object> call) {
        return new Param(key, Kind.BITS, rows, cols, call);
    }

    static Param close(String key, int rows, int cols, Function<Ctx, Object> call) {
        return new Param(key, Kind.CLOSE, rows, cols, call);
    }

    static Param permutation(String key, int rows, int cols, Function<Ctx, Object> call) {
        return new Param(key, Kind.PERMUTATION, rows, cols, call);
    }

    static Param write(String key, int rows, int cols, Function<Ctx, Object> call) {
        return new Param(key, Kind.WRITE, rows, cols, call);
    }

    // the argument under test plus the deterministic matrices a call creates
    static final class Ctx {
        final MatrixF x;
        private final List<MatrixF> created = new ArrayList<>();
        private final List<Supplier<MatrixF>> fresh = new ArrayList<>();
        private final List<Statistics.MomentsF> moments = new ArrayList<>();

        Ctx(MatrixF x) {
            this.x = x;
        }

        MatrixF t(int rows, int cols) {
            long seed = SEED * 31 + created.size();
            return track(() -> random(rows, cols, seed));
        }

        // a well-conditioned square matrix
        MatrixF sq(int n) {
            long seed = SEED * 37 + created.size();
            return track(() -> random(n, n, seed).addInplace(10.0f * n, Matrices.identityF(n)));
        }

        Statistics.MomentsF moments() {
            Statistics.MomentsF m = new Statistics.MomentsF();
            moments.add(m);
            return m;
        }

        MatrixF track(Supplier<MatrixF> s) {
            MatrixF m = s.get();
            created.add(m);
            fresh.add(s);
            return m;
        }

        void assertUntouched(String what) {
            for (int i = 0; i < created.size(); ++i) {
                assertBitsArray(what + ": argument " + i, fresh.get(i).get().getArrayUnsafe(),
                        created.get(i).getArrayUnsafe());
            }
            for (Statistics.MomentsF m : moments) {
                assertEquals(what + ": means", null, m.means);
                assertEquals(what + ": variances", null, m.variances);
            }
        }
    }

    // ---------------------------------------------------------------- helpers

    static boolean hasMatrixFParameter(Executable e) {
        for (Class<?> t : e.getParameterTypes()) {
            if (t == MatrixF.class || t == MatrixF[].class || t == Statistics.MomentsF.class) {
                return true;
            }
        }
        return false;
    }

    static void collect(Set<String> keys, String owner, List<Executable> members) {
        for (Executable e : members) {
            if (!Modifier.isPublic(e.getModifiers()) || e.isSynthetic()) {
                continue;
            }
            Class<?>[] types = e.getParameterTypes();
            for (int i = 0; i < types.length; ++i) {
                Class<?> t = types[i];
                if (t == MatrixF.class || t == MatrixF[].class || t == Statistics.MomentsF.class) {
                    keys.add(key(owner, e, i));
                }
            }
        }
    }

    static String key(String owner, Executable e, int index) {
        StringBuilder b = new StringBuilder(owner).append('.');
        b.append(e instanceof Constructor ? "<init>" : e.getName()).append('(');
        Class<?>[] types = e.getParameterTypes();
        for (int i = 0; i < types.length; ++i) {
            b.append(i == 0 ? "" : ",").append(types[i].getSimpleName());
        }
        return b.append(")#").append(index).toString();
    }

    static int[] s(int... rowsCols) {
        return rowsCols;
    }

    // a prefilled output, the same for every call
    static MatrixF filled(int rows, int cols) {
        return random(rows, cols, SEED - 1);
    }

    static void assertRefused(String what, Statistics.MomentsF m, Runnable call) {
        MatrixF means = m.means;
        MatrixF variances = m.variances;
        try {
            call.run();
            fail(what + ": no exception");
        } catch (UnsupportedOperationException expected) {
            // refused
        }
        assertSame(what + ": means", means, m.means);
        assertSame(what + ": variances", variances, m.variances);
    }

    static void assertClose(String what, MatrixF want, MatrixF got) {
        assertEquals(what + ": rows", want.numRows(), got.numRows());
        assertEquals(what + ": cols", want.numColumns(), got.numColumns());
        float[] w = want.getArrayUnsafe();
        float[] g = got.getArrayUnsafe();
        for (int i = 0; i < w.length; ++i) {
            float tol = 1.0e-4f * Math.max(1.0f, Math.max(Math.abs(w[i]), Math.abs(g[i])));
            if (!(Math.abs(w[i] - g[i]) <= tol)) {
                fail(what + " [" + i + "]: " + w[i] + " != " + g[i]);
            }
        }
    }

    // the result, or the class of the runtime exception thrown
    static Object outcome(Supplier<Object> call) {
        try {
            return call.get();
        } catch (RuntimeException e) {
            return e.getClass();
        }
    }

    static byte[] serialize(MatrixF m) {
        try {
            ByteArrayOutputStream os = new ByteArrayOutputStream();
            long n = Matrices.serializeF(m, os);
            assertEquals("bytes written", os.size(), n);
            return os.toByteArray();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static byte[] serializeToFile(MatrixF m) {
        Path file = null;
        try {
            file = Files.createTempFile("jamu-view", ".bin");
            Matrices.serializeF(m, file);
            return Files.readAllBytes(file);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } finally {
            if (file != null) {
                file.toFile().delete();
            }
        }
    }

    static TensorF layers(int rows, int cols, int depth, long seed) {
        TensorF t = new TensorF(rows, cols, depth);
        for (int layer = 0; layer < depth; ++layer) {
            t.set(random(rows, cols, seed + layer), layer);
        }
        return t;
    }

    // values in [1, 2) so that division never sees a zero
    static MatrixF random(int rows, int cols, long seed) {
        return Matrices.randomUniformF(rows, cols, 1.0f, 2.0f, seed);
    }

    static void assertResult(String what, Object want, Object got) {
        if (want instanceof MatrixF) {
            if (!(got instanceof MatrixF)) {
                fail(what + ": got " + got);
            }
            MatrixF w = (MatrixF) want;
            MatrixF g = (MatrixF) got;
            assertEquals(what + ": rows", w.numRows(), g.numRows());
            assertEquals(what + ": cols", w.numColumns(), g.numColumns());
            assertBitsArray(what, w.getArrayUnsafe(), g.getArrayUnsafe());
        } else if (want instanceof MatrixD) {
            MatrixD w = (MatrixD) want;
            MatrixD g = (MatrixD) got;
            assertEquals(what + ": rows", w.numRows(), g.numRows());
            assertEquals(what + ": cols", w.numColumns(), g.numColumns());
            double[] dw = w.getArrayUnsafe();
            double[] dg = g.getArrayUnsafe();
            assertEquals(what + ": length", dw.length, dg.length);
            for (int i = 0; i < dw.length; ++i) {
                assertEquals(what + " [" + i + "]", Double.doubleToRawLongBits(dw[i]),
                        Double.doubleToRawLongBits(dg[i]));
            }
        } else if (want instanceof ComplexMatrixF) {
            ComplexMatrixF w = (ComplexMatrixF) want;
            ComplexMatrixF g = (ComplexMatrixF) got;
            assertEquals(what + ": rows", w.numRows(), g.numRows());
            assertEquals(what + ": cols", w.numColumns(), g.numColumns());
            assertBitsArray(what, w.getArrayUnsafe(), g.getArrayUnsafe());
        } else if (want instanceof TensorF) {
            TensorF w = (TensorF) want;
            TensorF g = (TensorF) got;
            assertEquals(what + ": depth", w.numDepth(), g.numDepth());
            assertBitsArray(what, w.getArrayUnsafe(), g.getArrayUnsafe());
        } else if (want instanceof byte[]) {
            assertArrayEquals(what, (byte[]) want, (byte[]) got);
        } else if (want instanceof Float) {
            assertBits(what, (Float) want, (Float) got);
        } else {
            assertEquals(what, want, got);
        }
    }

    static void assertClose(String what, ComplexMatrixF want, ComplexMatrixF got) {
        assertEquals(what + ": rows", want.numRows(), got.numRows());
        assertEquals(what + ": cols", want.numColumns(), got.numColumns());
        float[] w = want.getArrayUnsafe();
        float[] g = got.getArrayUnsafe();
        for (int i = 0; i < w.length; ++i) {
            float tol = 1.0e-4f * Math.max(1.0f, Math.max(Math.abs(w[i]), Math.abs(g[i])));
            if (!(Math.abs(w[i] - g[i]) <= tol)) {
                fail(what + " [" + i + "]: " + w[i] + " != " + g[i]);
            }
        }
    }

    static void assertBits(String what, float want, float got) {
        assertEquals(what, Float.floatToRawIntBits(want), Float.floatToRawIntBits(got));
    }

    static void assertBitsArray(String what, float[] want, float[] got) {
        assertEquals(what + ": length", want.length, got.length);
        for (int i = 0; i < want.length; ++i) {
            assertBits(what + " [" + i + "]", want[i], got[i]);
        }
    }
}
