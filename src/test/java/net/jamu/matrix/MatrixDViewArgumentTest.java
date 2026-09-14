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
 * Tests for a {@code MatrixD} view passed as an argument.
 */
public final class MatrixDViewArgumentTest {

    private static final long SEED = 20260915L;

    @Test
    public void testArrayOfAnOrdinaryMatrixIsItsBackingArray() {
        MatrixD A = random(7, 4, SEED);
        assertSame(A.getArrayUnsafe(), ReadAccess.array(A));
    }

    @Test
    public void testArrayOfAViewIsAFreshCopyOfTheRegion() {
        MatrixD P = random(9, 8, SEED);
        double[] before = P.getArrayUnsafe().clone();
        MatrixD V = Matrices.view(P, 2, 1, 6, 5);
        double[] first = ReadAccess.array(V);
        double[] second = ReadAccess.array(V);
        assertNotSame(P.getArrayUnsafe(), first);
        assertNotSame(first, second);
        assertBitsArray("first", V.copy().getArrayUnsafe(), first);
        assertBitsArray("second", V.copy().getArrayUnsafe(), second);
        Arrays.fill(first, 12345.0);
        assertBitsArray("parent", before, P.getArrayUnsafe());
    }

    @Test
    public void testMatrixDBaseReadsAViewArgumentLikeItsCopy() {
        MatrixD P = random(60, 12, SEED + 1);
        double[] parent = P.getArrayUnsafe().clone();
        ArgCase[] cases = {
                new ArgCase("addInplace(B)", 6, 5, 6, 5, (T, B) -> T.addInplace(B)),
                new ArgCase("addInplace(alpha, B)", 6, 5, 6, 5, (T, B) -> T.addInplace(-1.5, B)),
                new ArgCase("add(B, C)", 6, 5, 6, 5, (T, B) -> T.add(B, Matrices.createD(6, 5))),
                new ArgCase("add(alpha, B, C)", 6, 5, 6, 5, (T, B) -> T.add(2.5, B, Matrices.createD(6, 5))),
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
                new ArgCase("setInplace(alpha, B)", 6, 5, 6, 5, (T, B) -> T.setInplace(3.0, B)),
                new ArgCase("setSubmatrixInplace height 10", 20, 8, 12, 6,
                        (T, B) -> T.setSubmatrixInplace(3, 1, B, 1, 1, 10, 5)),
                new ArgCase("setSubmatrixInplace height 45", 50, 8, 48, 6,
                        (T, B) -> T.setSubmatrixInplace(2, 1, B, 1, 1, 45, 5)),
                new ArgCase("setColumnInplace", 6, 5, 6, 1, (T, B) -> T.setColumnInplace(2, B)),
                new ArgCase("setInplaceUpperTrapezoidal", 6, 5, 7, 6, (T, B) -> T.setInplaceUpperTrapezoidal(B)),
                new ArgCase("setInplaceLowerTrapezoidal", 6, 5, 7, 6, (T, B) -> T.setInplaceLowerTrapezoidal(B)),
                new ArgCase("hadamard(B, out)", 6, 5, 6, 5, (T, B) -> T.hadamard(B, Matrices.createD(6, 5))),
                new ArgCase("hadamard(B)", 6, 5, 6, 5, (T, B) -> T.hadamard(B)),
                new ArgCase("appendColumn", 6, 5, 6, 1, (T, B) -> T.appendColumn(B)),
                new ArgCase("appendMatrix", 6, 5, 6, 3, (T, B) -> T.appendMatrix(B)),
                new ArgCase("hadamardTransposed", 6, 5, 5, 6, (T, B) -> T.hadamardTransposed(B)),
                new ArgCase("transposedHadamard", 6, 5, 5, 6, (T, B) -> T.transposedHadamard(B)) };
        Set<String> names = new HashSet<>();
        for (ArgCase c : cases) {
            assertEquals("duplicate " + c.name, true, names.add(c.name));
            MatrixD V = Matrices.view(P, 3, 2, 3 + c.argRows - 1, 2 + c.argCols - 1);
            MatrixD T1 = random(c.rows, c.cols, SEED + 2);
            MatrixD T2 = random(c.rows, c.cols, SEED + 2);
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
                new Overlap("addInplace", 50, 8, 0, 0, 49, 7, (A, B) -> A.addInplace(0.5, B)),
                new Overlap("setInplace", 50, 8, 0, 0, 49, 7, (A, B) -> A.setInplace(-2.0, B)),
                new Overlap("hadamard into the parent", 50, 8, 0, 0, 49, 7, (A, B) -> A.hadamard(B, A)),
                new Overlap("setInplaceUpperTrapezoidal", 50, 8, 2, 0, 49, 7,
                        (A, B) -> A.setInplaceUpperTrapezoidal(B)),
                new Overlap("setInplaceLowerTrapezoidal", 6, 20, 0, 3, 5, 19,
                        (A, B) -> A.setInplaceLowerTrapezoidal(B)) };
        for (Overlap c : cases) {
            MatrixD A1 = random(c.rows, c.cols, SEED + 3);
            MatrixD A2 = random(c.rows, c.cols, SEED + 3);
            MatrixD before = Matrices.view(A2, c.r0, c.c0, c.r1, c.c1).copy();
            c.call.accept(A1, Matrices.view(A1, c.r0, c.c0, c.r1, c.c1));
            c.call.accept(A2, before);
            assertBitsArray(c.name, A2.getArrayUnsafe(), A1.getArrayUnsafe());
        }
    }

    @Test
    public void testMatricesReadsAViewArgumentLikeItsCopy() {
        MatrixD P = random(20, 12, SEED + 4);
        P.set(17, 3, Double.NaN);
        double[] parent = P.getArrayUnsafe().clone();
        int[] v = { 3, 2, 8, 6 };
        int[] w = { 10, 4, 15, 8 };
        int[] row = { 5, 1, 5, 9 };
        int[] col = { 2, 7, 12, 7 };
        int[] nan = { 16, 2, 18, 4 };
        MatrixD O = random(6, 5, SEED + 5);
        MatrixD near = Matrices.view(P, v[0], v[1], v[2], v[3]).copy();
        near.set(4, 3, Math.nextUp(near.get(4, 3)));
        ComplexMatrixD Z = Matrices.randomUniformComplexD(4, 6, SEED + 6);
        Stat[] cases = {
                new Stat("serializeD(OutputStream)", true, v, v, (a, b) -> serialize(a)),
                new Stat("serializeD(Path)", true, v, v, (a, b) -> serializeToFile(a)),
                new Stat("convert", true, v, v, (a, b) -> Matrices.convert(a)),
                new Stat("convertToComplex", true, v, v, (a, b) -> Matrices.convertToComplex(a)),
                new Stat("ComplexMatrixD.times(MatrixD)", false, v, v, (a, b) -> Z.times(a)),
                new Stat("distance left", true, v, v, (a, b) -> Matrices.distance(a, O)),
                new Stat("distance right", true, v, v, (a, b) -> Matrices.distance(O, a)),
                new Stat("distance both", true, v, w, (a, b) -> Matrices.distance(a, b)),
                new Stat("approxEqual left", true, v, v, (a, b) -> Matrices.approxEqual(a, near)),
                new Stat("approxEqual right", true, v, v, (a, b) -> Matrices.approxEqual(near, a)),
                new Stat("approxEqual both", true, v, w, (a, b) -> Matrices.approxEqual(a, b)),
                new Stat("approxEqual relTol", true, v, v, (a, b) -> Matrices.approxEqual(a, near, 1.0e-17)),
                new Stat("approxEqual absTol", true, v, v, (a, b) -> Matrices.approxEqual(near, a, 0.0, 1.0e-12)),
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
            MatrixD a = Matrices.view(P, c.a[0], c.a[1], c.a[2], c.a[3]);
            MatrixD b = Matrices.view(P, c.b[0], c.b[1], c.b[2], c.b[3]);
            Object got = outcome(() -> c.call.apply(a, b));
            Object want = outcome(() -> c.call.apply(a.copy(), b.copy()));
            if (c.exact) {
                assertResult(c.name, want, got);
            } else {
                assertClose(c.name, (ComplexMatrixD) want, (ComplexMatrixD) got);
            }
            assertBitsArray(c.name + ": parent", parent, P.getArrayUnsafe());
        }
        assertEquals(IllegalArgumentException.class,
                outcome(() -> Matrices.round(Matrices.view(P, nan[0], nan[1], nan[2], nan[3]), 3)));
        assertEquals(true, Matrices.approxEqual(Matrices.view(P, v[0], v[1], v[2], v[3]), near));
        assertEquals(false, Matrices.approxEqual(Matrices.view(P, v[0], v[1], v[2], v[3]), near, 1.0e-17));
    }

    @Test
    public void testTensorReadsAViewArgumentLikeItsCopy() {
        MatrixD P = random(15, 11, SEED + 7);
        double[] parent = P.getArrayUnsafe().clone();
        MatrixD V = Matrices.view(P, 4, 3, 9, 7);
        MatrixD C = V.copy();

        assertBitsArray("TensorD(MatrixD)", new TensorD(C).getArrayUnsafe(), new TensorD(V).getArrayUnsafe());

        TensorD t1 = layers(6, 5, 3, SEED + 8);
        TensorD t2 = layers(6, 5, 3, SEED + 8);
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
        MatrixD P = random(40, 30, SEED + 9);
        double[] parent = P.getArrayUnsafe().clone();
        MatrixD A = random(6, 5, SEED + 10);
        double[] target = A.getArrayUnsafe().clone();
        Set<String> names = new HashSet<>();
        for (Prod c : products()) {
            assertEquals("duplicate " + c.name, true, names.add(c.name));
            MatrixD[] views = new MatrixD[c.shapes.length / 2];
            MatrixD[] copies = new MatrixD[views.length];
            for (int i = 0; i < views.length; ++i) {
                views[i] = productView(P, c, i);
                copies[i] = views[i].copy();
            }
            MatrixD got = c.call.apply(A, views);
            MatrixD want = c.call.apply(A, copies);
            assertClose(c.name, want, got);
            assertBitsArray(c.name + ": target", target, A.getArrayUnsafe());
            assertBitsArray(c.name + ": parent", parent, P.getArrayUnsafe());
        }
    }

    @Test
    public void testOperandOfAnOrdinaryMatrix() {
        MatrixD A = random(7, 4, SEED);
        ReadAccess.OperandD op = ReadAccess.operand(A, null);
        assertSame(A.getArrayUnsafe(), op.array);
        assertEquals(0, op.offset);
        assertEquals(7, op.ld);
        MatrixD R = random(1, 5, SEED);
        assertEquals(1, ReadAccess.operand(R, R.getArrayUnsafe()).ld);
        assertSame(R.getArrayUnsafe(), ReadAccess.operand(R, R.getArrayUnsafe()).array);
    }

    @Test
    public void testOperandOfAViewReadsTheParent() {
        MatrixD P = random(9, 8, SEED);
        MatrixD V = Matrices.view(P, 2, 3, 6, 5);
        ReadAccess.OperandD op = ReadAccess.operand(V, new double[1]);
        assertSame(P.getArrayUnsafe(), op.array);
        assertEquals(3 * 9 + 2, op.offset);
        assertEquals(9, op.ld);

        ReadAccess.OperandD inner = ReadAccess.operand(Matrices.view(V, 1, 1, 3, 2), null);
        assertSame(P.getArrayUnsafe(), inner.array);
        assertEquals(4 * 9 + 3, inner.offset);
        assertEquals(9, inner.ld);

        ReadAccess.OperandD onOut = ReadAccess.operand(V, P.getArrayUnsafe());
        assertNotSame(P.getArrayUnsafe(), onOut.array);
        assertEquals(0, onOut.offset);
        assertEquals(5, onOut.ld);
        assertBitsArray("copy", V.copy().getArrayUnsafe(), onOut.array);
    }

    @Test
    public void testViewArgumentProductsDoNotCopy() {
        MatrixD A = random(6, 5, SEED + 10);
        for (Prod c : products()) {
            CountingMatrixD P = counting(40, 30, SEED + 9);
            MatrixD[] views = new MatrixD[c.shapes.length / 2];
            for (int i = 0; i < views.length; ++i) {
                views[i] = productView(P, c, i);
            }
            MatrixD got = c.call.apply(A, views);
            // the addend of timesPlus and timesMinus is copied into the result
            int addend = c.name.startsWith("timesPlus") || c.name.startsWith("timesMinus") ? 1 : 0;
            assertEquals(c.name + ": copies", addend, P.copies);
            MatrixD[] copies = new MatrixD[views.length];
            for (int i = 0; i < views.length; ++i) {
                copies[i] = views[i].copy();
            }
            assertClose(c.name, c.call.apply(A, copies), got);
        }

        // large enough for the threaded kernels
        CountingMatrixD P = counting(400, 300, SEED + 21);
        MatrixD L = random(250, 180, SEED + 22);
        MatrixD B = Matrices.view(P, 70, 50, 249, 209);
        MatrixD Bt = Matrices.view(P, 70, 50, 319, 229);
        MatrixD Bb = Matrices.view(P, 20, 40, 199, 219);
        MatrixD[] got = { L.mult(B, Matrices.createD(250, 160)), L.transAmultAdd(Bt, filled(180, 180)),
                L.transBmult(Bb, Matrices.createD(250, 180)), L.times(B) };
        assertEquals("large: copies", 0, P.copies);
        MatrixD[] want = { L.mult(B.copy(), Matrices.createD(250, 160)),
                L.transAmultAdd(Bt.copy(), filled(180, 180)), L.transBmult(Bb.copy(), Matrices.createD(250, 180)),
                L.times(B.copy()) };
        for (int i = 0; i < want.length; ++i) {
            assertClose("large " + i, want[i], got[i]);
        }
    }

    @Test
    public void testViewArgumentOnTheOutputIsCopiedOnce() {
        MatrixD A = random(6, 5, SEED + 11);
        CountingMatrixD C = counting(6, 4, SEED + 12);
        MatrixD C2 = random(6, 4, SEED + 12);
        MatrixD before = Matrices.view(C2, 1, 0, 5, 3).copy();
        A.multAdd(0.5, Matrices.view(C, 1, 0, 5, 3), C);
        assertEquals("copies", 1, C.copies);
        A.multAdd(0.5, before, C2);
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
                CountingMatrixD PA = counting(40, 30, SEED + 23);
                CountingMatrixD PB = sameParent ? PA : counting(40, 30, SEED + 24);
                MatrixD V = Matrices.view(PA, 7, 6, 12, 10);
                MatrixD[] views = new MatrixD[c.shapes.length / 2];
                for (int i = 0; i < views.length; ++i) {
                    views[i] = productView(PB, c, i);
                }
                MatrixD got = c.call.apply(V, views);
                int addend = c.name.startsWith("timesPlus") || c.name.startsWith("timesMinus") ? 1 : 0;
                assertEquals(what + ": copies", addend, PA.copies + (sameParent ? 0 : PB.copies));
                MatrixD[] copies = new MatrixD[views.length];
                for (int i = 0; i < views.length; ++i) {
                    copies[i] = views[i].copy();
                }
                assertClose(what, c.call.apply(V.copy(), copies), got);
            }
        }

        CountingMatrixD S = counting(12, 11, SEED + 25);
        MatrixD W = Matrices.view(S, 2, 3, 8, 7);
        MatrixD[] got = { W.timesTransposed(), W.transposedTimes() };
        assertEquals("timesTransposed and transposedTimes: copies", 0, S.copies);
        assertClose("timesTransposed()", W.copy().timesTransposed(), got[0]);
        assertClose("transposedTimes()", W.copy().transposedTimes(), got[1]);

        CountingMatrixD P = counting(400, 300, SEED + 26);
        MatrixD L = Matrices.view(P, 70, 50, 319, 229);
        MatrixD B = random(180, 160, SEED + 27);
        MatrixD[] large = { L.times(B), L.transAmultAdd(Matrices.view(P, 10, 5, 259, 144), filled(180, 140)),
                L.timesTransposed() };
        assertEquals("large: copies", 0, P.copies);
        MatrixD Lc = L.copy();
        assertClose("large times", Lc.times(B), large[0]);
        assertClose("large transAmultAdd",
                Lc.transAmultAdd(Matrices.view(P, 10, 5, 259, 144).copy(), filled(180, 140)), large[1]);
        assertClose("large timesTransposed", Lc.timesTransposed(), large[2]);
    }

    @Test
    public void testViewProductsIntoTheParent() {
        // receiver region within C, receiver shape, argument shape, output shape
        ParentOut[] cases = {
                new ParentOut("multAdd", 0, 1, 6, 4, 4, 5, 6, 5, (V, B, C) -> V.multAdd(-1.0, B, C)),
                new ParentOut("mult", 0, 1, 6, 4, 4, 5, 6, 5, (V, B, C) -> V.mult(2.0, B, C)),
                new ParentOut("transAmultAdd", 1, 0, 4, 5, 4, 6, 5, 6, (V, B, C) -> V.transAmultAdd(B, C)),
                new ParentOut("transAmult", 1, 0, 4, 5, 4, 6, 5, 6, (V, B, C) -> V.transAmult(B, C)),
                new ParentOut("transBmultAdd", 0, 1, 5, 3, 4, 3, 5, 4, (V, B, C) -> V.transBmultAdd(0.5, B, C)),
                new ParentOut("transBmult", 0, 1, 5, 3, 4, 3, 5, 4, (V, B, C) -> V.transBmult(B, C)),
                new ParentOut("transABmultAdd", 1, 1, 4, 5, 6, 4, 5, 6, (V, B, C) -> V.transABmultAdd(B, C)),
                new ParentOut("transABmult", 1, 1, 4, 5, 6, 4, 5, 6, (V, B, C) -> V.transABmult(-3.0, B, C)) };
        for (ParentOut c : cases) {
            MatrixD B = random(c.argRows, c.argCols, SEED + 28);
            CountingMatrixD C1 = counting(c.outRows, c.outCols, SEED + 29);
            MatrixD C2 = random(c.outRows, c.outCols, SEED + 29);
            MatrixD before = Matrices.view(C2, c.r0, c.c0, c.r0 + c.rows - 1, c.c0 + c.cols - 1).copy();
            MatrixD V = Matrices.view(C1, c.r0, c.c0, c.r0 + c.rows - 1, c.c0 + c.cols - 1);
            assertSame(c.name, C1, c.call.apply(V, B, C1));
            assertEquals(c.name + ": copies", 1, C1.copies);
            c.call.apply(before, B, C2);
            assertClose(c.name, C2, C1);
        }

        // the argument, not the receiver, is on the output
        MatrixD V = Matrices.view(random(9, 9, SEED + 30), 2, 2, 7, 5);
        CountingMatrixD C1 = counting(6, 5, SEED + 31);
        MatrixD C2 = random(6, 5, SEED + 31);
        MatrixD before = Matrices.view(C2, 1, 0, 4, 4).copy();
        V.multAdd(Matrices.view(C1, 1, 0, 4, 4), C1);
        assertEquals("argument on the output: copies", 1, C1.copies);
        V.copy().multAdd(before, C2);
        assertClose("argument on the output", C2, C1);
        CountingMatrixD D1 = counting(6, 5, SEED + 31);
        MatrixD D2 = random(6, 5, SEED + 31);
        MatrixD beforeD = Matrices.view(D2, 1, 0, 4, 4).copy();
        V.mult(Matrices.view(D1, 1, 0, 4, 4), D1);
        assertEquals("argument on the zeroed output: copies", 1, D1.copies);
        V.copy().mult(beforeD, D2);
        assertClose("argument on the zeroed output", D2, D1);
    }

    @Test
    public void testTimesManyStillCopiesOnce() {
        CountingMatrixD P = counting(10, 9, SEED + 32);
        MatrixD V = Matrices.view(P, 2, 1, 7, 5);
        MatrixD B = random(5, 4, SEED + 33);
        MatrixD C = random(4, 3, SEED + 34);
        MatrixD D = random(3, 2, SEED + 35);
        MatrixD got = V.timesMany(B, C, D);
        assertEquals("copies", 1, P.copies);
        assertClose("timesMany", V.copy().timesMany(B, C, D), got);
    }

    static Prod[] products() {
        return new Prod[] {
                new Prod("mult", s(5, 4), (T, x) -> T.mult(x[0], Matrices.createD(6, 4))),
                new Prod("mult alpha", s(5, 4), (T, x) -> T.mult(-0.5, x[0], Matrices.createD(6, 4))),
                new Prod("multAdd", s(5, 4), (T, x) -> T.multAdd(x[0], filled(6, 4))),
                new Prod("multAdd alpha", s(5, 4), (T, x) -> T.multAdd(1.5, x[0], filled(6, 4))),
                new Prod("transABmult", s(4, 6), (T, x) -> T.transABmult(x[0], Matrices.createD(5, 4))),
                new Prod("transABmult alpha", s(4, 6), (T, x) -> T.transABmult(2.0, x[0], Matrices.createD(5, 4))),
                new Prod("transABmultAdd", s(4, 6), (T, x) -> T.transABmultAdd(x[0], filled(5, 4))),
                new Prod("transABmultAdd alpha", s(4, 6), (T, x) -> T.transABmultAdd(-1.0, x[0], filled(5, 4))),
                new Prod("transAmult", s(6, 4), (T, x) -> T.transAmult(x[0], Matrices.createD(5, 4))),
                new Prod("transAmult alpha", s(6, 4), (T, x) -> T.transAmult(3.0, x[0], Matrices.createD(5, 4))),
                new Prod("transAmultAdd", s(6, 4), (T, x) -> T.transAmultAdd(x[0], filled(5, 4))),
                new Prod("transAmultAdd alpha", s(6, 4), (T, x) -> T.transAmultAdd(0.25, x[0], filled(5, 4))),
                new Prod("transBmult", s(4, 5), (T, x) -> T.transBmult(x[0], Matrices.createD(6, 4))),
                new Prod("transBmult alpha", s(4, 5), (T, x) -> T.transBmult(-2.0, x[0], Matrices.createD(6, 4))),
                new Prod("transBmultAdd", s(4, 5), (T, x) -> T.transBmultAdd(x[0], filled(6, 4))),
                new Prod("transBmultAdd alpha", s(4, 5), (T, x) -> T.transBmultAdd(0.5, x[0], filled(6, 4))),
                new Prod("times", s(5, 4), (T, x) -> T.times(x[0])),
                new Prod("timesTimes", s(5, 4, 4, 3), (T, x) -> T.timesTimes(x[0], x[1])),
                new Prod("timesMany", s(5, 4, 4, 3, 3, 2), (T, x) -> T.timesMany(x[0], x[1], x[2])),
                new Prod("timesTransposed", s(3, 5), (T, x) -> T.timesTransposed(x[0])),
                new Prod("transposedTimes", s(6, 3), (T, x) -> T.transposedTimes(x[0])),
                new Prod("timesPlus", s(5, 4, 6, 4), (T, x) -> T.timesPlus(x[0], x[1])),
                new Prod("timesMinus", s(5, 4, 6, 4), (T, x) -> T.timesMinus(x[0], x[1])) };
    }

    // the i-th argument of a product case, at an offset that differs per argument
    static MatrixD productView(MatrixD P, Prod c, int i) {
        int r0 = 1 + 3 * i;
        int c0 = 2 + 2 * i;
        return Matrices.view(P, r0, c0, r0 + c.shapes[2 * i] - 1, c0 + c.shapes[2 * i + 1] - 1);
    }

    @Test
    public void testProductWithTheOutputBeingTheParentOfTheArgument() {
        MatrixD A = random(6, 5, SEED + 11);
        MatrixD C1 = random(6, 4, SEED + 12);
        MatrixD C2 = random(6, 4, SEED + 12);
        MatrixD before = Matrices.view(C2, 1, 0, 5, 3).copy();
        assertSame(C1, A.multAdd(-1.0, Matrices.view(C1, 1, 0, 5, 3), C1));
        A.multAdd(-1.0, before, C2);
        assertClose("multAdd", C2, C1);

        MatrixD At = random(4, 5, SEED + 13);
        MatrixD D1 = random(5, 4, SEED + 14);
        MatrixD D2 = random(5, 4, SEED + 14);
        MatrixD beforeT = Matrices.view(D2, 1, 0, 4, 3).copy();
        assertSame(D1, At.transAmultAdd(2.0, Matrices.view(D1, 1, 0, 4, 3), D1));
        At.transAmultAdd(2.0, beforeT, D2);
        assertClose("transAmultAdd", D2, D1);
    }

    @Test
    public void testZeroedOutputIsNotReadThroughAView() {
        // A shape, C shape, B region within C
        OutputAlias[] cases = {
                new OutputAlias("mult", 6, 5, 6, 4, 1, 0, 5, 3, (A, B, C) -> A.mult(B, C)),
                new OutputAlias("mult alpha", 6, 5, 6, 4, 1, 0, 5, 3, (A, B, C) -> A.mult(-2.0, B, C)),
                new OutputAlias("transAmult", 4, 5, 5, 4, 1, 0, 4, 3, (A, B, C) -> A.transAmult(B, C)),
                new OutputAlias("transAmult alpha", 4, 5, 5, 4, 1, 0, 4, 3, (A, B, C) -> A.transAmult(0.5, B, C)),
                new OutputAlias("transBmult", 7, 5, 7, 6, 1, 0, 6, 4, (A, B, C) -> A.transBmult(B, C)),
                new OutputAlias("transBmult alpha", 7, 5, 7, 6, 1, 0, 6, 4, (A, B, C) -> A.transBmult(3.0, B, C)),
                new OutputAlias("transABmult", 4, 6, 6, 5, 1, 1, 5, 4, (A, B, C) -> A.transABmult(B, C)),
                new OutputAlias("transABmult alpha", 4, 6, 6, 5, 1, 1, 5, 4,
                        (A, B, C) -> A.transABmult(-1.0, B, C)) };
        for (OutputAlias c : cases) {
            MatrixD A = random(c.rows, c.cols, SEED + 19);
            MatrixD C1 = random(c.outRows, c.outCols, SEED + 20);
            MatrixD C2 = random(c.outRows, c.outCols, SEED + 20);
            MatrixD before = Matrices.view(C2, c.r0, c.c0, c.r1, c.c1).copy();
            assertSame(c.name, C1, c.call.apply(A, Matrices.view(C1, c.r0, c.c0, c.r1, c.c1), C1));
            c.call.apply(A, before, C2);
            assertClose(c.name, C2, C1);
        }
    }

    @Test
    public void testMomentsViewIsRefusedBeforeAIsWritten() {
        MatrixD Pr = random(3, 9, SEED + 15);
        MatrixD Pc = random(12, 4, SEED + 16);
        double[] rowParent = Pr.getArrayUnsafe().clone();
        double[] colParent = Pc.getArrayUnsafe().clone();
        MatrixD A = random(8, 5, SEED + 17);
        double[] a = A.getArrayUnsafe().clone();
        Statistics.MomentsD[] columns = {
                new Statistics.MomentsD(Matrices.view(Pr, 1, 2, 1, 6), Matrices.createD(1, 5)),
                new Statistics.MomentsD(Matrices.createD(1, 5), Matrices.view(Pr, 2, 0, 2, 4)),
                new Statistics.MomentsD(Matrices.view(Pr, 0, 0, 1, 2), null) };
        Statistics.MomentsD[] rows = {
                new Statistics.MomentsD(Matrices.view(Pc, 2, 1, 9, 1), Matrices.createD(8, 1)),
                new Statistics.MomentsD(Matrices.createD(8, 1), Matrices.view(Pc, 0, 3, 7, 3)),
                new Statistics.MomentsD(null, Matrices.view(Pc, 0, 0, 2, 2)) };
        for (int i = 0; i < columns.length; ++i) {
            Statistics.MomentsD m = columns[i];
            assertRefused("zscoreColumnsInplace " + i, m, () -> Statistics.zscoreColumnsInplace(A, m));
        }
        for (int i = 0; i < rows.length; ++i) {
            Statistics.MomentsD m = rows[i];
            assertRefused("zscoreRowsInplace " + i, m, () -> Statistics.zscoreRowsInplace(A, m));
        }
        assertBitsArray("A", a, A.getArrayUnsafe());
        assertBitsArray("row parent", rowParent, Pr.getArrayUnsafe());
        assertBitsArray("column parent", colParent, Pc.getArrayUnsafe());
    }

    @Test
    public void testEveryMatrixDParameterIsClassified() {
        // implementations must not add public MatrixD parameters of their own
        for (Class<?> impl : new Class<?>[] { MatrixDBase.class, SimpleMatrixD.class }) {
            for (Method m : impl.getMethods()) {
                if (hasMatrixDParameter(m)) {
                    try {
                        MatrixD.class.getMethod(m.getName(), m.getParameterTypes());
                    } catch (NoSuchMethodException e) {
                        fail(impl.getSimpleName() + " declares " + key(impl.getSimpleName(), m, -1));
                    }
                }
            }
        }

        Set<String> found = new TreeSet<>();
        collect(found, "MatrixD", Arrays.asList(MatrixD.class.getMethods()));
        collect(found, "ComplexMatrixD", Arrays.asList(ComplexMatrixD.class.getMethods()));
        for (Class<?> c : new Class<?>[] { Matrices.class, Statistics.class, TensorD.class }) {
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

        MatrixD P = random(60, 40, SEED + 18).addInplace(30.0, Matrices.identityD(60).selectConsecutiveColumns(0, 39));
        double[] parent = P.getArrayUnsafe().clone();
        for (Param p : table) {
            MatrixD V = Matrices.view(P, 4, 4, 4 + p.rows - 1, 4 + p.cols - 1);
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
                    if (want instanceof ComplexMatrixD) {
                        assertClose(p.key, (ComplexMatrixD) want, (ComplexMatrixD) got);
                    } else {
                        assertClose(p.key, (MatrixD) want, (MatrixD) got);
                    }
                } else {
                    double[] w = ((MatrixD) want).getArrayUnsafe().clone();
                    double[] g = ((MatrixD) got).getArrayUnsafe().clone();
                    Arrays.sort(w);
                    Arrays.sort(g);
                    assertBitsArray(p.key, w, g);
                }
            }
            assertBitsArray(p.key + ": parent", parent, P.getArrayUnsafe());
        }
    }

    // every (member, MatrixD parameter) pair; x is the view or its copy
    static Param[] params() {
        String B = "(MatrixD,MatrixD)#";
        String aB = "(double,MatrixD,MatrixD)#";
        return new Param[] {
                // MatrixD: products read B and write C
                close("MatrixD.mult" + B + 0, 5, 4, c -> c.t(6, 5).mult(c.x, c.t(6, 4))),
                close("MatrixD.mult" + aB + 1, 5, 4, c -> c.t(6, 5).mult(0.5, c.x, c.t(6, 4))),
                close("MatrixD.multAdd" + B + 0, 5, 4, c -> c.t(6, 5).multAdd(c.x, c.t(6, 4))),
                close("MatrixD.multAdd" + aB + 1, 5, 4, c -> c.t(6, 5).multAdd(2.0, c.x, c.t(6, 4))),
                close("MatrixD.transAmult" + B + 0, 6, 4, c -> c.t(6, 5).transAmult(c.x, c.t(5, 4))),
                close("MatrixD.transAmult" + aB + 1, 6, 4, c -> c.t(6, 5).transAmult(0.5, c.x, c.t(5, 4))),
                close("MatrixD.transAmultAdd" + B + 0, 6, 4, c -> c.t(6, 5).transAmultAdd(c.x, c.t(5, 4))),
                close("MatrixD.transAmultAdd" + aB + 1, 6, 4,
                        c -> c.t(6, 5).transAmultAdd(2.0, c.x, c.t(5, 4))),
                close("MatrixD.transBmult" + B + 0, 4, 5, c -> c.t(6, 5).transBmult(c.x, c.t(6, 4))),
                close("MatrixD.transBmult" + aB + 1, 4, 5, c -> c.t(6, 5).transBmult(0.5, c.x, c.t(6, 4))),
                close("MatrixD.transBmultAdd" + B + 0, 4, 5, c -> c.t(6, 5).transBmultAdd(c.x, c.t(6, 4))),
                close("MatrixD.transBmultAdd" + aB + 1, 4, 5,
                        c -> c.t(6, 5).transBmultAdd(2.0, c.x, c.t(6, 4))),
                close("MatrixD.transABmult" + B + 0, 4, 6, c -> c.t(6, 5).transABmult(c.x, c.t(5, 4))),
                close("MatrixD.transABmult" + aB + 1, 4, 6, c -> c.t(6, 5).transABmult(0.5, c.x, c.t(5, 4))),
                close("MatrixD.transABmultAdd" + B + 0, 4, 6, c -> c.t(6, 5).transABmultAdd(c.x, c.t(5, 4))),
                close("MatrixD.transABmultAdd" + aB + 1, 4, 6,
                        c -> c.t(6, 5).transABmultAdd(2.0, c.x, c.t(5, 4))),
                write("MatrixD.mult" + B + 1, 6, 4, c -> c.t(6, 5).mult(c.t(5, 4), c.x)),
                write("MatrixD.mult" + aB + 2, 6, 4, c -> c.t(6, 5).mult(0.5, c.t(5, 4), c.x)),
                write("MatrixD.multAdd" + B + 1, 6, 4, c -> c.t(6, 5).multAdd(c.t(5, 4), c.x)),
                write("MatrixD.multAdd" + aB + 2, 6, 4, c -> c.t(6, 5).multAdd(2.0, c.t(5, 4), c.x)),
                write("MatrixD.transAmult" + B + 1, 5, 4, c -> c.t(6, 5).transAmult(c.t(6, 4), c.x)),
                write("MatrixD.transAmult" + aB + 2, 5, 4, c -> c.t(6, 5).transAmult(0.5, c.t(6, 4), c.x)),
                write("MatrixD.transAmultAdd" + B + 1, 5, 4, c -> c.t(6, 5).transAmultAdd(c.t(6, 4), c.x)),
                write("MatrixD.transAmultAdd" + aB + 2, 5, 4,
                        c -> c.t(6, 5).transAmultAdd(2.0, c.t(6, 4), c.x)),
                write("MatrixD.transBmult" + B + 1, 6, 4, c -> c.t(6, 5).transBmult(c.t(4, 5), c.x)),
                write("MatrixD.transBmult" + aB + 2, 6, 4, c -> c.t(6, 5).transBmult(0.5, c.t(4, 5), c.x)),
                write("MatrixD.transBmultAdd" + B + 1, 6, 4, c -> c.t(6, 5).transBmultAdd(c.t(4, 5), c.x)),
                write("MatrixD.transBmultAdd" + aB + 2, 6, 4,
                        c -> c.t(6, 5).transBmultAdd(2.0, c.t(4, 5), c.x)),
                write("MatrixD.transABmult" + B + 1, 5, 4, c -> c.t(6, 5).transABmult(c.t(4, 6), c.x)),
                write("MatrixD.transABmult" + aB + 2, 5, 4, c -> c.t(6, 5).transABmult(0.5, c.t(4, 6), c.x)),
                write("MatrixD.transABmultAdd" + B + 1, 5, 4, c -> c.t(6, 5).transABmultAdd(c.t(4, 6), c.x)),
                write("MatrixD.transABmultAdd" + aB + 2, 5, 4,
                        c -> c.t(6, 5).transABmultAdd(2.0, c.t(4, 6), c.x)),
                // MatrixD: element-wise reads, in place on the receiver
                bits("MatrixD.add" + B + 0, 6, 5, c -> c.t(6, 5).add(c.x, c.t(6, 5))),
                bits("MatrixD.add" + aB + 1, 6, 5, c -> c.t(6, 5).add(-1.5, c.x, c.t(6, 5))),
                write("MatrixD.add" + B + 1, 6, 5, c -> c.t(6, 5).add(c.t(6, 5), c.x)),
                write("MatrixD.add" + aB + 2, 6, 5, c -> c.t(6, 5).add(-1.5, c.t(6, 5), c.x)),
                bits("MatrixD.addInplace(MatrixD)#0", 6, 5, c -> c.t(6, 5).addInplace(c.x)),
                bits("MatrixD.addInplace(double,MatrixD)#1", 6, 5, c -> c.t(6, 5).addInplace(0.5, c.x)),
                bits("MatrixD.addBroadcastedVectorInplace(MatrixD)#0", 6, 1,
                        c -> c.t(6, 5).addBroadcastedVectorInplace(c.x)),
                bits("MatrixD.mulBroadcastedVectorInplace(MatrixD)#0", 6, 1,
                        c -> c.t(6, 5).mulBroadcastedVectorInplace(c.x)),
                bits("MatrixD.divBroadcastedVectorInplace(MatrixD)#0", 6, 1,
                        c -> c.t(6, 5).divBroadcastedVectorInplace(c.x)),
                bits("MatrixD.addBroadcastedRowVectorInplace(MatrixD)#0", 1, 5,
                        c -> c.t(6, 5).addBroadcastedRowVectorInplace(c.x)),
                bits("MatrixD.mulBroadcastedRowVectorInplace(MatrixD)#0", 1, 5,
                        c -> c.t(6, 5).mulBroadcastedRowVectorInplace(c.x)),
                bits("MatrixD.divBroadcastedRowVectorInplace(MatrixD)#0", 1, 5,
                        c -> c.t(6, 5).divBroadcastedRowVectorInplace(c.x)),
                bits("MatrixD.setSubmatrixInplace(int,int,MatrixD,int,int,int,int)#2", 10, 4,
                        c -> c.t(12, 6).setSubmatrixInplace(1, 1, c.x, 0, 0, 9, 3)),
                bits("MatrixD.setInplaceUpperTrapezoidal(MatrixD)#0", 7, 6,
                        c -> c.t(6, 5).setInplaceUpperTrapezoidal(c.x)),
                bits("MatrixD.setInplaceLowerTrapezoidal(MatrixD)#0", 7, 6,
                        c -> c.t(6, 5).setInplaceLowerTrapezoidal(c.x)),
                bits("MatrixD.setColumnInplace(int,MatrixD)#1", 6, 1, c -> c.t(6, 5).setColumnInplace(2, c.x)),
                bits("MatrixD.setInplace(MatrixD)#0", 6, 5, c -> c.t(6, 5).setInplace(c.x)),
                bits("MatrixD.setInplace(double,MatrixD)#1", 6, 5, c -> c.t(6, 5).setInplace(3.0, c.x)),
                bits("MatrixD.hadamard" + B + 0, 6, 5, c -> c.t(6, 5).hadamard(c.x, c.t(6, 5))),
                write("MatrixD.hadamard" + B + 1, 6, 5, c -> c.t(6, 5).hadamard(c.t(6, 5), c.x)),
                close("MatrixD.solve" + B + 0, 5, 2, c -> c.sq(5).solve(c.x, c.t(5, 2))),
                write("MatrixD.solve" + B + 1, 5, 2, c -> c.sq(5).solve(c.t(5, 2), c.x)),
                write("MatrixD.trans(MatrixD)#0", 5, 6, c -> c.t(6, 5).trans(c.x)),
                write("MatrixD.scale(double,MatrixD)#1", 6, 5, c -> c.t(6, 5).scale(2.0, c.x)),
                write("MatrixD.inv(MatrixD)#0", 5, 5, c -> c.sq(5).inv(c.x)),
                write("MatrixD.submatrix(int,int,int,int,MatrixD,int,int)#4", 3, 3,
                        c -> c.t(6, 5).submatrix(0, 0, 2, 2, c.x, 0, 0)),
                // MatrixD: reads that allocate their result
                bits("MatrixD.plusBroadcastedVector(MatrixD)#0", 6, 1, c -> c.t(6, 5).plusBroadcastedVector(c.x)),
                bits("MatrixD.mulBroadcastedVector(MatrixD)#0", 6, 1, c -> c.t(6, 5).mulBroadcastedVector(c.x)),
                bits("MatrixD.divBroadcastedVector(MatrixD)#0", 6, 1, c -> c.t(6, 5).divBroadcastedVector(c.x)),
                bits("MatrixD.plusBroadcastedRowVector(MatrixD)#0", 1, 5,
                        c -> c.t(6, 5).plusBroadcastedRowVector(c.x)),
                bits("MatrixD.mulBroadcastedRowVector(MatrixD)#0", 1, 5,
                        c -> c.t(6, 5).mulBroadcastedRowVector(c.x)),
                bits("MatrixD.divBroadcastedRowVector(MatrixD)#0", 1, 5,
                        c -> c.t(6, 5).divBroadcastedRowVector(c.x)),
                bits("MatrixD.hadamardTransposed(MatrixD)#0", 5, 6, c -> c.t(6, 5).hadamardTransposed(c.x)),
                bits("MatrixD.transposedHadamard(MatrixD)#0", 5, 6, c -> c.t(6, 5).transposedHadamard(c.x)),
                bits("MatrixD.plus(MatrixD)#0", 6, 5, c -> c.t(6, 5).plus(c.x)),
                bits("MatrixD.minus(MatrixD)#0", 6, 5, c -> c.t(6, 5).minus(c.x)),
                bits("MatrixD.hadamard(MatrixD)#0", 6, 5, c -> c.t(6, 5).hadamard(c.x)),
                bits("MatrixD.appendColumn(MatrixD)#0", 6, 1, c -> c.t(6, 5).appendColumn(c.x)),
                bits("MatrixD.appendMatrix(MatrixD)#0", 6, 3, c -> c.t(6, 5).appendMatrix(c.x)),
                close("MatrixD.times(MatrixD)#0", 5, 4, c -> c.t(6, 5).times(c.x)),
                close("MatrixD.timesTimes" + B + 0, 5, 4, c -> c.t(6, 5).timesTimes(c.x, c.t(4, 3))),
                close("MatrixD.timesTimes" + B + 1, 4, 3, c -> c.t(6, 5).timesTimes(c.t(5, 4), c.x)),
                close("MatrixD.timesMany(MatrixD,MatrixD[])#0", 5, 4,
                        c -> c.t(6, 5).timesMany(c.x, c.t(4, 3), c.t(3, 2))),
                close("MatrixD.timesMany(MatrixD,MatrixD[])#1", 4, 3,
                        c -> c.t(6, 5).timesMany(c.t(5, 4), c.x, c.t(3, 2))),
                close("MatrixD.timesPlus" + B + 0, 5, 4, c -> c.t(6, 5).timesPlus(c.x, c.t(6, 4))),
                close("MatrixD.timesPlus" + B + 1, 6, 4, c -> c.t(6, 5).timesPlus(c.t(5, 4), c.x)),
                close("MatrixD.timesMinus" + B + 0, 5, 4, c -> c.t(6, 5).timesMinus(c.x, c.t(6, 4))),
                close("MatrixD.timesMinus" + B + 1, 6, 4, c -> c.t(6, 5).timesMinus(c.t(5, 4), c.x)),
                close("MatrixD.timesTransposed(MatrixD)#0", 3, 5, c -> c.t(6, 5).timesTransposed(c.x)),
                close("MatrixD.transposedTimes(MatrixD)#0", 6, 3, c -> c.t(6, 5).transposedTimes(c.x)),
                close("MatrixD.mldivide(MatrixD)#0", 5, 2, c -> c.sq(5).mldivide(c.x)),
                close("MatrixD.mrdivide(MatrixD)#0", 5, 5, c -> c.t(4, 5).mrdivide(c.x)),
                close("ComplexMatrixD.times(MatrixD)#0", 6, 5,
                        c -> Matrices.randomUniformComplexD(4, 6, SEED).times(c.x)),
                // Matrices
                bits("Matrices.view(MatrixD,int,int,int,int)#0", 6, 5, c -> Matrices.view(c.x, 1, 1, 4, 3).copy()),
                bits("Matrices.embed(int,int,MatrixD)#2", 6, 5, c -> Matrices.embed(8, 7, c.x)),
                bits("Matrices.sameDimD(MatrixD)#0", 6, 5, c -> Matrices.sameDimD(c.x)),
                bits("Matrices.serializeD(MatrixD,Path)#0", 6, 5, c -> serializeToFile(c.x)),
                bits("Matrices.serializeD(MatrixD,OutputStream)#0", 6, 5, c -> serialize(c.x)),
                bits("Matrices.convert(MatrixD)#0", 6, 5, c -> Matrices.convert(c.x)),
                bits("Matrices.convertToComplex(MatrixD)#0", 6, 5, c -> Matrices.convertToComplex(c.x)),
                bits("Matrices.round(MatrixD,int)#0", 6, 5, c -> Matrices.round(c.x, 4)),
                bits("Matrices.timeDelayEmbeddingD(MatrixD,int)#0", 6, 5, c -> Matrices.timeDelayEmbeddingD(c.x, 2)),
                bits("Matrices.sumRows(MatrixD)#0", 6, 5, c -> Matrices.sumRows(c.x)),
                bits("Matrices.sumColumns(MatrixD)#0", 6, 5, c -> Matrices.sumColumns(c.x)),
                bits("Matrices.rowsAverage(MatrixD)#0", 6, 5, c -> Matrices.rowsAverage(c.x)),
                bits("Matrices.colsAverage(MatrixD)#0", 6, 5, c -> Matrices.colsAverage(c.x)),
                bits("Matrices.distance(MatrixD,MatrixD)#0", 6, 5, c -> Matrices.distance(c.x, c.t(6, 5))),
                bits("Matrices.distance(MatrixD,MatrixD)#1", 6, 5, c -> Matrices.distance(c.t(6, 5), c.x)),
                bits("Matrices.approxEqual(MatrixD,MatrixD)#0", 6, 5, c -> Matrices.approxEqual(c.x, c.x.copy())),
                bits("Matrices.approxEqual(MatrixD,MatrixD)#1", 6, 5, c -> Matrices.approxEqual(c.t(6, 5), c.x)),
                bits("Matrices.approxEqual(MatrixD,MatrixD,double)#0", 6, 5,
                        c -> Matrices.approxEqual(c.x, c.x.copy(), 1.0e-3)),
                bits("Matrices.approxEqual(MatrixD,MatrixD,double)#1", 6, 5,
                        c -> Matrices.approxEqual(c.t(6, 5), c.x, 1.0)),
                bits("Matrices.approxEqual(MatrixD,MatrixD,double,double)#0", 6, 5,
                        c -> Matrices.approxEqual(c.x, c.t(6, 5), 0.0, 1.0)),
                bits("Matrices.approxEqual(MatrixD,MatrixD,double,double)#1", 6, 5,
                        c -> Matrices.approxEqual(c.t(6, 5), c.x, 0.0, 0.1)),
                bits("Matrices.numericalRank(MatrixD)#0", 6, 5, c -> Matrices.numericalRank(c.x)),
                bits("Matrices.numericalRank(MatrixD,double)#0", 6, 5, c -> Matrices.numericalRank(c.x, 1.0)),
                // Statistics
                bits("Statistics.centerColumns(MatrixD)#0", 6, 5, c -> Statistics.centerColumns(c.x)),
                bits("Statistics.zscoreColumns(MatrixD)#0", 6, 5, c -> Statistics.zscoreColumns(c.x)),
                bits("Statistics.zscoreRows(MatrixD)#0", 6, 5, c -> Statistics.zscoreRows(c.x)),
                bits("Statistics.rescale(MatrixD,double,double)#0", 6, 5, c -> Statistics.rescale(c.x, -1.0, 1.0)),
                bits("Statistics.shuffleColumns(MatrixD,long)#0", 6, 5, c -> Statistics.shuffleColumns(c.x, 7L)),
                bits("Statistics.shuffleRows(MatrixD,long)#0", 6, 5, c -> Statistics.shuffleRows(c.x, 7L)),
                permutation("Statistics.shuffleColumns(MatrixD)#0", 6, 5, c -> Statistics.shuffleColumns(c.x)),
                permutation("Statistics.shuffleRows(MatrixD)#0", 6, 5, c -> Statistics.shuffleRows(c.x)),
                write("Statistics.centerColumnsInplace(MatrixD)#0", 6, 5, c -> Statistics.centerColumnsInplace(c.x)),
                write("Statistics.zscoreColumnsInplace(MatrixD)#0", 6, 5, c -> Statistics.zscoreColumnsInplace(c.x)),
                write("Statistics.zscoreRowsInplace(MatrixD)#0", 6, 5, c -> Statistics.zscoreRowsInplace(c.x)),
                write("Statistics.zscoreColumnsInplace(MatrixD,MomentsD)#0", 6, 5,
                        c -> Statistics.zscoreColumnsInplace(c.x, c.moments())),
                write("Statistics.zscoreRowsInplace(MatrixD,MomentsD)#0", 6, 5,
                        c -> Statistics.zscoreRowsInplace(c.x, c.moments())),
                write("Statistics.zscoreColumnsInplace(MatrixD,MomentsD)#1", 1, 5,
                        c -> Statistics.zscoreColumnsInplace(c.t(8, 5), new Statistics.MomentsD(c.x, null))),
                write("Statistics.zscoreRowsInplace(MatrixD,MomentsD)#1", 8, 1,
                        c -> Statistics.zscoreRowsInplace(c.t(8, 5), new Statistics.MomentsD(null, c.x))),
                write("Statistics.rescaleInplace(MatrixD,double,double)#0", 6, 5,
                        c -> Statistics.rescaleInplace(c.x, 0.0, 1.0)),
                write("Statistics.shuffleColumnsInplace(MatrixD)#0", 6, 5, c -> Statistics.shuffleColumnsInplace(c.x)),
                write("Statistics.shuffleColumnsInplace(MatrixD,long)#0", 6, 5,
                        c -> Statistics.shuffleColumnsInplace(c.x, 7L)),
                write("Statistics.shuffleRowsInplace(MatrixD)#0", 6, 5, c -> Statistics.shuffleRowsInplace(c.x)),
                write("Statistics.shuffleRowsInplace(MatrixD,long)#0", 6, 5,
                        c -> Statistics.shuffleRowsInplace(c.x, 7L)),
                // TensorD
                bits("TensorD.<init>(MatrixD)#0", 6, 5, c -> new TensorD(c.x)),
                bits("TensorD.set(MatrixD,int)#0", 6, 5, c -> layers(6, 5, 3, SEED).set(c.x, 1)),
                bits("TensorD.append(MatrixD)#0", 6, 5, c -> layers(6, 5, 3, SEED).append(c.x)) };
    }

    // ---------------------------------------------------------------- cases

    static final class ArgCase {
        final String name;
        final int rows;
        final int cols;
        final int argRows;
        final int argCols;
        final BiFunction<MatrixD, MatrixD, Object> call;

        ArgCase(String name, int rows, int cols, int argRows, int argCols, BiFunction<MatrixD, MatrixD, Object> call) {
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
        final BiConsumer<MatrixD, MatrixD> call;

        Overlap(String name, int rows, int cols, int r0, int c0, int r1, int c1, BiConsumer<MatrixD, MatrixD> call) {
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
        final BiFunction<MatrixD, MatrixD, Object> call;

        Stat(String name, boolean exact, int[] a, int[] b, BiFunction<MatrixD, MatrixD, Object> call) {
            this.name = name;
            this.exact = exact;
            this.a = a;
            this.b = b;
            this.call = call;
        }
    }

    // counts the block copies a view of this matrix makes
    static final class CountingMatrixD extends SimpleMatrixD {
        int copies;

        CountingMatrixD(int rows, int cols) {
            super(rows, cols);
        }

        @Override
        public MatrixD submatrix(int r0, int c0, int r1, int c1, MatrixD B, int rb, int cb) {
            ++copies;
            return super.submatrix(r0, c0, r1, c1, B, rb, cb);
        }
    }

    static CountingMatrixD counting(int rows, int cols, long seed) {
        CountingMatrixD m = new CountingMatrixD(rows, cols);
        m.setInplace(random(rows, cols, seed));
        return m;
    }

    interface TriFunction {
        MatrixD apply(MatrixD A, MatrixD B, MatrixD C);
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
        final BiFunction<MatrixD, MatrixD[], MatrixD> call;

        Prod(String name, int[] shapes, BiFunction<MatrixD, MatrixD[], MatrixD> call) {
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
        final MatrixD x;
        private final List<MatrixD> created = new ArrayList<>();
        private final List<Supplier<MatrixD>> fresh = new ArrayList<>();
        private final List<Statistics.MomentsD> moments = new ArrayList<>();

        Ctx(MatrixD x) {
            this.x = x;
        }

        MatrixD t(int rows, int cols) {
            long seed = SEED * 31 + created.size();
            return track(() -> random(rows, cols, seed));
        }

        // a well-conditioned square matrix
        MatrixD sq(int n) {
            long seed = SEED * 37 + created.size();
            return track(() -> random(n, n, seed).addInplace(10.0 * n, Matrices.identityD(n)));
        }

        Statistics.MomentsD moments() {
            Statistics.MomentsD m = new Statistics.MomentsD();
            moments.add(m);
            return m;
        }

        MatrixD track(Supplier<MatrixD> s) {
            MatrixD m = s.get();
            created.add(m);
            fresh.add(s);
            return m;
        }

        void assertUntouched(String what) {
            for (int i = 0; i < created.size(); ++i) {
                assertBitsArray(what + ": argument " + i, fresh.get(i).get().getArrayUnsafe(),
                        created.get(i).getArrayUnsafe());
            }
            for (Statistics.MomentsD m : moments) {
                assertEquals(what + ": means", null, m.means);
                assertEquals(what + ": variances", null, m.variances);
            }
        }
    }

    // ---------------------------------------------------------------- helpers

    static boolean hasMatrixDParameter(Executable e) {
        for (Class<?> t : e.getParameterTypes()) {
            if (t == MatrixD.class || t == MatrixD[].class || t == Statistics.MomentsD.class) {
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
                if (t == MatrixD.class || t == MatrixD[].class || t == Statistics.MomentsD.class) {
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
    static MatrixD filled(int rows, int cols) {
        return random(rows, cols, SEED - 1);
    }

    static void assertRefused(String what, Statistics.MomentsD m, Runnable call) {
        MatrixD means = m.means;
        MatrixD variances = m.variances;
        try {
            call.run();
            fail(what + ": no exception");
        } catch (UnsupportedOperationException expected) {
            // refused
        }
        assertSame(what + ": means", means, m.means);
        assertSame(what + ": variances", variances, m.variances);
    }

    static void assertClose(String what, MatrixD want, MatrixD got) {
        assertEquals(what + ": rows", want.numRows(), got.numRows());
        assertEquals(what + ": cols", want.numColumns(), got.numColumns());
        double[] w = want.getArrayUnsafe();
        double[] g = got.getArrayUnsafe();
        for (int i = 0; i < w.length; ++i) {
            double tol = 1.0e-10 * Math.max(1.0, Math.max(Math.abs(w[i]), Math.abs(g[i])));
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

    static byte[] serialize(MatrixD m) {
        try {
            ByteArrayOutputStream os = new ByteArrayOutputStream();
            long n = Matrices.serializeD(m, os);
            assertEquals("bytes written", os.size(), n);
            return os.toByteArray();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static byte[] serializeToFile(MatrixD m) {
        Path file = null;
        try {
            file = Files.createTempFile("jamu-view", ".bin");
            Matrices.serializeD(m, file);
            return Files.readAllBytes(file);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } finally {
            if (file != null) {
                file.toFile().delete();
            }
        }
    }

    static TensorD layers(int rows, int cols, int depth, long seed) {
        TensorD t = new TensorD(rows, cols, depth);
        for (int layer = 0; layer < depth; ++layer) {
            t.set(random(rows, cols, seed + layer), layer);
        }
        return t;
    }

    // values in [1, 2) so that division never sees a zero
    static MatrixD random(int rows, int cols, long seed) {
        return Matrices.randomUniformD(rows, cols, 1.0, 2.0, seed);
    }

    static void assertResult(String what, Object want, Object got) {
        if (want instanceof MatrixD) {
            if (!(got instanceof MatrixD)) {
                fail(what + ": got " + got);
            }
            MatrixD w = (MatrixD) want;
            MatrixD g = (MatrixD) got;
            assertEquals(what + ": rows", w.numRows(), g.numRows());
            assertEquals(what + ": cols", w.numColumns(), g.numColumns());
            assertBitsArray(what, w.getArrayUnsafe(), g.getArrayUnsafe());
        } else if (want instanceof MatrixF) {
            MatrixF w = (MatrixF) want;
            MatrixF g = (MatrixF) got;
            assertEquals(what + ": rows", w.numRows(), g.numRows());
            assertEquals(what + ": cols", w.numColumns(), g.numColumns());
            float[] fw = w.getArrayUnsafe();
            float[] fg = g.getArrayUnsafe();
            assertEquals(what + ": length", fw.length, fg.length);
            for (int i = 0; i < fw.length; ++i) {
                assertEquals(what + " [" + i + "]", Float.floatToRawIntBits(fw[i]), Float.floatToRawIntBits(fg[i]));
            }
        } else if (want instanceof ComplexMatrixD) {
            ComplexMatrixD w = (ComplexMatrixD) want;
            ComplexMatrixD g = (ComplexMatrixD) got;
            assertEquals(what + ": rows", w.numRows(), g.numRows());
            assertEquals(what + ": cols", w.numColumns(), g.numColumns());
            assertBitsArray(what, w.getArrayUnsafe(), g.getArrayUnsafe());
        } else if (want instanceof TensorD) {
            TensorD w = (TensorD) want;
            TensorD g = (TensorD) got;
            assertEquals(what + ": depth", w.numDepth(), g.numDepth());
            assertBitsArray(what, w.getArrayUnsafe(), g.getArrayUnsafe());
        } else if (want instanceof byte[]) {
            assertArrayEquals(what, (byte[]) want, (byte[]) got);
        } else if (want instanceof Double) {
            assertBits(what, (Double) want, (Double) got);
        } else {
            assertEquals(what, want, got);
        }
    }

    static void assertClose(String what, ComplexMatrixD want, ComplexMatrixD got) {
        assertEquals(what + ": rows", want.numRows(), got.numRows());
        assertEquals(what + ": cols", want.numColumns(), got.numColumns());
        double[] w = want.getArrayUnsafe();
        double[] g = got.getArrayUnsafe();
        for (int i = 0; i < w.length; ++i) {
            double tol = 1.0e-10 * Math.max(1.0, Math.max(Math.abs(w[i]), Math.abs(g[i])));
            if (!(Math.abs(w[i] - g[i]) <= tol)) {
                fail(what + " [" + i + "]: " + w[i] + " != " + g[i]);
            }
        }
    }

    static void assertBits(String what, double want, double got) {
        assertEquals(what, Double.doubleToRawLongBits(want), Double.doubleToRawLongBits(got));
    }

    static void assertBitsArray(String what, double[] want, double[] got) {
        assertEquals(what + ": length", want.length, got.length);
        for (int i = 0; i < want.length; ++i) {
            assertBits(what + " [" + i + "]", want[i], got[i]);
        }
    }
}
