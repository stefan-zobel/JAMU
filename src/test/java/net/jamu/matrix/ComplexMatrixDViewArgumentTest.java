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
 * Tests for a {@code ComplexMatrixD} view passed as an argument.
 */
public final class ComplexMatrixDViewArgumentTest {

    private static final long SEED = 20260917L;

    @Test
    public void testArrayOfAnOrdinaryMatrixIsItsBackingArray() {
        ComplexMatrixD A = random(7, 4, SEED);
        assertSame(A.getArrayUnsafe(), ReadAccess.array(A));
    }

    @Test
    public void testArrayOfAViewIsAFreshCopyOfTheRegion() {
        ComplexMatrixD P = random(9, 8, SEED);
        double[] before = P.getArrayUnsafe().clone();
        for (ComplexMatrixD V : new ComplexMatrixD[] { Matrices.view(P, 2, 1, 6, 5), Matrices.view(P, 0, 0, 4, 4) }) {
            double[] first = ReadAccess.array(V);
            double[] second = ReadAccess.array(V);
            assertNotSame(P.getArrayUnsafe(), first);
            assertNotSame(first, second);
            assertBitsArray("first", V.copy().getArrayUnsafe(), first);
            assertBitsArray("second", V.copy().getArrayUnsafe(), second);
            Arrays.fill(first, 12345.0);
            assertBitsArray("parent", before, P.getArrayUnsafe());
        }
    }

    @Test
    public void testComplexMatrixDBaseReadsAViewArgumentLikeItsCopy() {
        ComplexMatrixD P = random(60, 12, SEED + 1);
        double[] parent = P.getArrayUnsafe().clone();
        ArgCase[] cases = {
                new ArgCase("addInplace(B)", 6, 5, 6, 5, (T, B) -> T.addInplace(B)),
                new ArgCase("addInplace(alpha, B)", 6, 5, 6, 5, (T, B) -> T.addInplace(-1.5, 0.5, B)),
                new ArgCase("add(B, C)", 6, 5, 6, 5, (T, B) -> T.add(B, Matrices.createComplexD(6, 5))),
                new ArgCase("add(alpha, B, C)", 6, 5, 6, 5,
                        (T, B) -> T.add(2.5, -1.0, B, Matrices.createComplexD(6, 5))),
                new ArgCase("plus(B)", 6, 5, 6, 5, (T, B) -> T.plus(B)),
                new ArgCase("minus(B)", 6, 5, 6, 5, (T, B) -> T.minus(B)),
                new ArgCase("setInplace(B)", 6, 5, 6, 5, (T, B) -> T.setInplace(B)),
                new ArgCase("setInplace(alpha, B)", 6, 5, 6, 5, (T, B) -> T.setInplace(3.0, -2.0, B)),
                new ArgCase("setSubmatrixInplace height 10", 20, 8, 12, 6,
                        (T, B) -> T.setSubmatrixInplace(3, 1, B, 1, 1, 10, 5)),
                new ArgCase("setSubmatrixInplace height 45", 50, 8, 48, 6,
                        (T, B) -> T.setSubmatrixInplace(2, 1, B, 1, 1, 45, 5)),
                new ArgCase("setColumnInplace", 6, 5, 6, 1, (T, B) -> T.setColumnInplace(2, B)),
                new ArgCase("setInplaceUpperTrapezoidal", 6, 5, 7, 6, (T, B) -> T.setInplaceUpperTrapezoidal(B)),
                new ArgCase("setInplaceLowerTrapezoidal", 6, 5, 7, 6, (T, B) -> T.setInplaceLowerTrapezoidal(B)),
                new ArgCase("hadamard(B, out)", 6, 5, 6, 5,
                        (T, B) -> T.hadamard(B, Matrices.createComplexD(6, 5))),
                new ArgCase("hadamard(B)", 6, 5, 6, 5, (T, B) -> T.hadamard(B)),
                new ArgCase("appendColumn", 6, 5, 6, 1, (T, B) -> T.appendColumn(B)),
                new ArgCase("appendMatrix", 6, 5, 6, 3, (T, B) -> T.appendMatrix(B)) };
        Set<String> names = new HashSet<>();
        for (ArgCase c : cases) {
            assertEquals("duplicate " + c.name, true, names.add(c.name));
            for (boolean anchored : new boolean[] { false, true }) {
                String what = c.name + (anchored ? " at (0, 0)" : "");
                int r0 = anchored ? 0 : 3;
                int c0 = anchored ? 0 : 2;
                ComplexMatrixD V = Matrices.view(P, r0, c0, r0 + c.argRows - 1, c0 + c.argCols - 1);
                ComplexMatrixD T1 = random(c.rows, c.cols, SEED + 2);
                ComplexMatrixD T2 = random(c.rows, c.cols, SEED + 2);
                Object got = c.call.apply(T1, V);
                Object want = c.call.apply(T2, V.copy());
                assertResult(what, want, got);
                assertBitsArray(what + ": target", T2.getArrayUnsafe(), T1.getArrayUnsafe());
                assertBitsArray(what + ": parent", parent, P.getArrayUnsafe());
            }
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
                new Overlap("addInplace", 50, 8, 0, 0, 49, 7, (A, B) -> A.addInplace(0.5, 0.25, B)),
                new Overlap("setInplace", 50, 8, 0, 0, 49, 7, (A, B) -> A.setInplace(-2.0, 1.0, B)),
                new Overlap("hadamard into the parent", 50, 8, 0, 0, 49, 7, (A, B) -> A.hadamard(B, A)),
                new Overlap("add into the parent", 50, 8, 0, 0, 49, 7, (A, B) -> A.add(B, A)),
                new Overlap("setInplaceUpperTrapezoidal", 50, 8, 2, 0, 49, 7,
                        (A, B) -> A.setInplaceUpperTrapezoidal(B)),
                new Overlap("setInplaceLowerTrapezoidal", 6, 20, 0, 3, 5, 19,
                        (A, B) -> A.setInplaceLowerTrapezoidal(B)) };
        for (Overlap c : cases) {
            ComplexMatrixD A1 = random(c.rows, c.cols, SEED + 3);
            ComplexMatrixD A2 = random(c.rows, c.cols, SEED + 3);
            ComplexMatrixD before = Matrices.view(A2, c.r0, c.c0, c.r1, c.c1).copy();
            c.call.accept(A1, Matrices.view(A1, c.r0, c.c0, c.r1, c.c1));
            c.call.accept(A2, before);
            assertBitsArray(c.name, A2.getArrayUnsafe(), A1.getArrayUnsafe());
        }
    }

    @Test
    public void testMatricesReadsAViewArgumentLikeItsCopy() {
        ComplexMatrixD P = random(20, 12, SEED + 4);
        P.set(17, 3, Double.NaN, 0.0);
        double[] parent = P.getArrayUnsafe().clone();
        int[] v = { 3, 2, 8, 6 };
        int[] w = { 10, 4, 15, 8 };
        int[] at0 = { 0, 0, 5, 4 };
        int[] row = { 5, 1, 5, 9 };
        int[] col = { 2, 7, 12, 7 };
        int[] nan = { 16, 2, 18, 4 };
        ComplexMatrixD O = random(6, 5, SEED + 5);
        ComplexMatrixD near = Matrices.view(P, v[0], v[1], v[2], v[3]).copy();
        near.set(4, 3, Math.nextUp(near.get(4, 3).re()), near.get(4, 3).im());
        MatrixD R = Matrices.randomUniformD(4, 6, SEED + 6);
        Stat[] cases = {
                new Stat("serializeComplexD(OutputStream)", true, v, v, (a, b) -> serialize(a)),
                new Stat("serializeComplexD(Path)", true, v, v, (a, b) -> serializeToFile(a)),
                new Stat("convert", true, v, v, (a, b) -> Matrices.convert(a)),
                new Stat("convertToReal", true, v, v, (a, b) -> Matrices.convertToReal(a)),
                new Stat("MatrixD.times(ComplexMatrixD)", false, v, v, (a, b) -> R.times(a)),
                new Stat("MatrixD.times(ComplexMatrixD) at (0, 0)", false, at0, at0, (a, b) -> R.times(a)),
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
                new Stat("sumColumns single column", true, col, col, (a, b) -> Matrices.sumColumns(a)) };
        Set<String> names = new HashSet<>();
        for (Stat c : cases) {
            assertEquals("duplicate " + c.name, true, names.add(c.name));
            ComplexMatrixD a = Matrices.view(P, c.a[0], c.a[1], c.a[2], c.a[3]);
            ComplexMatrixD b = Matrices.view(P, c.b[0], c.b[1], c.b[2], c.b[3]);
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
    public void testStatisticsRefusesAViewBeforeWriting() {
        ComplexMatrixD P = random(9, 8, SEED + 36);
        double[] parent = P.getArrayUnsafe().clone();
        for (ComplexMatrixD V : new ComplexMatrixD[] { Matrices.view(P, 1, 2, 7, 6), Matrices.view(P, 0, 0, 6, 4) }) {
            assertEquals(UnsupportedOperationException.class, outcome(() -> Statistics.centerColumnsInplace(V)));
            assertEquals(UnsupportedOperationException.class, outcome(() -> Statistics.zscoreColumnsInplace(V)));
            assertBitsArray("parent", parent, P.getArrayUnsafe());
            assertResult("centerColumns", Statistics.centerColumns(V.copy()), Statistics.centerColumns(V));
            assertResult("zscoreColumns", Statistics.zscoreColumns(V.copy()), Statistics.zscoreColumns(V));
            assertBitsArray("parent after the reads", parent, P.getArrayUnsafe());
        }
    }

    @Test
    public void testProductsReadAViewArgumentLikeItsCopy() {
        ComplexMatrixD P = random(40, 30, SEED + 9);
        double[] parent = P.getArrayUnsafe().clone();
        ComplexMatrixD A = random(6, 5, SEED + 10);
        double[] target = A.getArrayUnsafe().clone();
        Set<String> names = new HashSet<>();
        for (Prod c : products()) {
            assertEquals("duplicate " + c.name, true, names.add(c.name));
            for (boolean anchored : new boolean[] { false, true }) {
                String what = c.name + (anchored ? " at (0, 0)" : "");
                ComplexMatrixD[] views = new ComplexMatrixD[c.shapes.length / 2];
                ComplexMatrixD[] copies = new ComplexMatrixD[views.length];
                for (int i = 0; i < views.length; ++i) {
                    views[i] = productView(P, c, i, anchored);
                    copies[i] = views[i].copy();
                }
                ComplexMatrixD got = c.call.apply(A, views);
                ComplexMatrixD want = c.call.apply(A, copies);
                assertClose(what, want, got);
                assertBitsArray(what + ": target", target, A.getArrayUnsafe());
                assertBitsArray(what + ": parent", parent, P.getArrayUnsafe());
            }
        }
    }

    @Test
    public void testOperandOfAnOrdinaryMatrix() {
        ComplexMatrixD A = random(7, 4, SEED);
        ReadAccess.OperandD op = ReadAccess.operand(A, null);
        assertSame(A.getArrayUnsafe(), op.array);
        assertEquals(0, op.offset);
        assertEquals(7, op.ld);
        ComplexMatrixD R = random(1, 5, SEED);
        assertEquals(1, ReadAccess.operand(R, R.getArrayUnsafe()).ld);
        assertSame(R.getArrayUnsafe(), ReadAccess.operand(R, R.getArrayUnsafe()).array);
    }

    @Test
    public void testOperandOfAViewAtTheOriginReadsTheParent() {
        ComplexMatrixD P = random(9, 8, SEED);
        ComplexMatrixD V = Matrices.view(P, 0, 0, 6, 5);
        ReadAccess.OperandD op = ReadAccess.operand(V, new double[1]);
        assertSame(P.getArrayUnsafe(), op.array);
        assertEquals(0, op.offset);
        assertEquals(9, op.ld);

        ReadAccess.OperandD inner = ReadAccess.operand(Matrices.view(V, 0, 0, 3, 2), null);
        assertSame(P.getArrayUnsafe(), inner.array);
        assertEquals(0, inner.offset);
        assertEquals(9, inner.ld);

        ReadAccess.OperandD onOut = ReadAccess.operand(V, P.getArrayUnsafe());
        assertNotSame(P.getArrayUnsafe(), onOut.array);
        assertEquals(0, onOut.offset);
        assertEquals(7, onOut.ld);
        assertBitsArray("copy on the output", V.copy().getArrayUnsafe(), onOut.array);
    }

    @Test
    public void testOperandOfAViewWithAnOffsetIsACopy() {
        ComplexMatrixD P = random(9, 8, SEED);
        int[][] regions = { { 2, 3, 6, 5 }, { 1, 0, 6, 5 }, { 0, 1, 6, 5 } };
        for (int[] reg : regions) {
            ComplexMatrixD V = Matrices.view(P, reg[0], reg[1], reg[2], reg[3]);
            ReadAccess.OperandD op = ReadAccess.operand(V, new double[1]);
            String what = Arrays.toString(reg);
            assertNotSame(what, P.getArrayUnsafe(), op.array);
            assertEquals(what, 0, op.offset);
            assertEquals(what, V.numRows(), op.ld);
            assertBitsArray(what, V.copy().getArrayUnsafe(), op.array);
        }
        // a view of a view at (0, 0) still has an offset in the parent
        ComplexMatrixD inner = Matrices.view(Matrices.view(P, 1, 1, 7, 6), 0, 0, 2, 2);
        assertNotSame(P.getArrayUnsafe(), ReadAccess.operand(inner, null).array);
    }

    @Test
    public void testViewArgumentProductsCopyOnlyWithAnOffset() {
        ComplexMatrixD A = random(6, 5, SEED + 10);
        for (Prod c : products()) {
            for (boolean anchored : new boolean[] { false, true }) {
                String what = c.name + (anchored ? " at (0, 0)" : "");
                CountingComplexMatrixD P = counting(40, 30, SEED + 9);
                ComplexMatrixD[] views = new ComplexMatrixD[c.shapes.length / 2];
                for (int i = 0; i < views.length; ++i) {
                    views[i] = productView(P, c, i, anchored);
                }
                ComplexMatrixD got = c.call.apply(A, views);
                // the addend of timesPlus and timesMinus is copied into the result
                int addend = c.name.startsWith("timesPlus") || c.name.startsWith("timesMinus") ? 1 : 0;
                assertEquals(what + ": copies", anchored ? addend : views.length, P.copies);
                ComplexMatrixD[] copies = new ComplexMatrixD[views.length];
                for (int i = 0; i < views.length; ++i) {
                    copies[i] = views[i].copy();
                }
                assertClose(what, c.call.apply(A, copies), got);
            }
        }

        // large enough for the threaded kernels
        CountingComplexMatrixD P = counting(400, 300, SEED + 21);
        ComplexMatrixD L = random(250, 180, SEED + 22);
        ComplexMatrixD B = Matrices.view(P, 0, 0, 179, 159);
        ComplexMatrixD Bt = Matrices.view(P, 0, 0, 249, 179);
        ComplexMatrixD Bb = Matrices.view(P, 0, 0, 199, 179);
        ComplexMatrixD[] got = { L.mult(B, Matrices.createComplexD(250, 160)),
                L.conjTransAmultAdd(Bt, filled(180, 180)), L.conjTransBmult(Bb, Matrices.createComplexD(250, 200)),
                L.times(B) };
        assertEquals("large: copies", 0, P.copies);
        ComplexMatrixD[] want = { L.mult(B.copy(), Matrices.createComplexD(250, 160)),
                L.conjTransAmultAdd(Bt.copy(), filled(180, 180)),
                L.conjTransBmult(Bb.copy(), Matrices.createComplexD(250, 200)), L.times(B.copy()) };
        for (int i = 0; i < want.length; ++i) {
            assertClose("large " + i, want[i], got[i]);
        }
    }

    @Test
    public void testViewArgumentOnTheOutputIsCopiedOnce() {
        for (boolean anchored : new boolean[] { false, true }) {
            int r0 = anchored ? 0 : 1;
            ComplexMatrixD A = random(6, 5, SEED + 11);
            CountingComplexMatrixD C = counting(6, 4, SEED + 12);
            ComplexMatrixD C2 = random(6, 4, SEED + 12);
            ComplexMatrixD before = Matrices.view(C2, r0, 0, r0 + 4, 3).copy();
            A.multAdd(0.5, -0.5, Matrices.view(C, r0, 0, r0 + 4, 3), C);
            assertEquals("copies", 1, C.copies);
            A.multAdd(0.5, -0.5, before, C2);
            assertClose("multAdd", C2, C);
        }
    }

    @Test
    public void testViewProductsCopyOnlyWithAnOffset() {
        for (Prod c : products()) {
            if (c.name.equals("timesMany")) {
                continue;
            }
            for (boolean anchored : new boolean[] { false, true }) {
                // the arguments on a second parent, or on the receiver's own parent
                for (boolean sameParent : new boolean[] { false, true }) {
                    String what = c.name + (anchored ? " at (0, 0)" : "") + (sameParent ? " (same parent)" : "");
                    CountingComplexMatrixD PA = counting(40, 30, SEED + 23);
                    CountingComplexMatrixD PB = sameParent ? PA : counting(40, 30, SEED + 24);
                    ComplexMatrixD V = anchored ? Matrices.view(PA, 0, 0, 5, 4) : Matrices.view(PA, 7, 6, 12, 10);
                    ComplexMatrixD[] views = new ComplexMatrixD[c.shapes.length / 2];
                    for (int i = 0; i < views.length; ++i) {
                        views[i] = productView(PB, c, i, anchored);
                    }
                    ComplexMatrixD got = c.call.apply(V, views);
                    int addend = c.name.startsWith("timesPlus") || c.name.startsWith("timesMinus") ? 1 : 0;
                    assertEquals(what + ": copies", anchored ? addend : 1 + views.length,
                            PA.copies + (sameParent ? 0 : PB.copies));
                    ComplexMatrixD[] copies = new ComplexMatrixD[views.length];
                    for (int i = 0; i < views.length; ++i) {
                        copies[i] = views[i].copy();
                    }
                    assertClose(what, c.call.apply(V.copy(), copies), got);
                }
            }
        }

        for (boolean anchored : new boolean[] { false, true }) {
            CountingComplexMatrixD S = counting(12, 11, SEED + 25);
            ComplexMatrixD W = anchored ? Matrices.view(S, 0, 0, 6, 4) : Matrices.view(S, 2, 3, 8, 7);
            ComplexMatrixD[] got = { W.timesConjugateTransposed(), W.conjugateTransposedTimes() };
            assertEquals("self products: copies", anchored ? 0 : 4, S.copies);
            assertClose("timesConjugateTransposed()", W.copy().timesConjugateTransposed(), got[0]);
            assertClose("conjugateTransposedTimes()", W.copy().conjugateTransposedTimes(), got[1]);
        }

        CountingComplexMatrixD P = counting(400, 300, SEED + 26);
        ComplexMatrixD L = Matrices.view(P, 0, 0, 249, 179);
        ComplexMatrixD B = random(180, 160, SEED + 27);
        ComplexMatrixD[] large = { L.times(B),
                L.conjTransAmultAdd(Matrices.view(P, 0, 0, 249, 139), filled(180, 140)),
                L.timesConjugateTransposed() };
        assertEquals("large: copies", 0, P.copies);
        ComplexMatrixD Lc = L.copy();
        assertClose("large times", Lc.times(B), large[0]);
        assertClose("large conjTransAmultAdd",
                Lc.conjTransAmultAdd(Matrices.view(P, 0, 0, 249, 139).copy(), filled(180, 140)), large[1]);
        assertClose("large timesConjugateTransposed", Lc.timesConjugateTransposed(), large[2]);
    }

    @Test
    public void testViewProductsIntoTheParent() {
        // receiver region within C (at the origin, so only the output check copies), receiver shape,
        // argument shape, output shape
        ParentOut[] cases = {
                new ParentOut("multAdd", 6, 4, 4, 5, 6, 5, (V, B, C) -> V.multAdd(-1.0, 0.5, B, C)),
                new ParentOut("mult", 6, 4, 4, 5, 6, 5, (V, B, C) -> V.mult(2.0, 0.0, B, C)),
                new ParentOut("conjTransAmultAdd", 4, 5, 4, 6, 5, 6, (V, B, C) -> V.conjTransAmultAdd(B, C)),
                new ParentOut("conjTransAmult", 4, 5, 4, 6, 5, 6, (V, B, C) -> V.conjTransAmult(B, C)),
                new ParentOut("conjTransBmultAdd", 5, 3, 4, 3, 5, 4,
                        (V, B, C) -> V.conjTransBmultAdd(0.5, 1.0, B, C)),
                new ParentOut("conjTransBmult", 5, 3, 4, 3, 5, 4, (V, B, C) -> V.conjTransBmult(B, C)),
                new ParentOut("conjTransABmultAdd", 4, 5, 6, 4, 5, 6, (V, B, C) -> V.conjTransABmultAdd(B, C)),
                new ParentOut("conjTransABmult", 4, 5, 6, 4, 5, 6,
                        (V, B, C) -> V.conjTransABmult(-3.0, 1.0, B, C)) };
        for (ParentOut c : cases) {
            ComplexMatrixD B = random(c.argRows, c.argCols, SEED + 28);
            CountingComplexMatrixD C1 = counting(c.outRows, c.outCols, SEED + 29);
            ComplexMatrixD C2 = random(c.outRows, c.outCols, SEED + 29);
            ComplexMatrixD before = Matrices.view(C2, 0, 0, c.rows - 1, c.cols - 1).copy();
            ComplexMatrixD V = Matrices.view(C1, 0, 0, c.rows - 1, c.cols - 1);
            assertSame(c.name, C1, c.call.apply(V, B, C1));
            assertEquals(c.name + ": copies", 1, C1.copies);
            c.call.apply(before, B, C2);
            assertClose(c.name, C2, C1);
        }

        // the argument, not the receiver, is on the output
        ComplexMatrixD V = Matrices.view(random(9, 9, SEED + 30), 0, 0, 5, 3);
        CountingComplexMatrixD C1 = counting(6, 5, SEED + 31);
        ComplexMatrixD C2 = random(6, 5, SEED + 31);
        ComplexMatrixD before = Matrices.view(C2, 0, 0, 3, 4).copy();
        V.multAdd(Matrices.view(C1, 0, 0, 3, 4), C1);
        assertEquals("argument on the output: copies", 1, C1.copies);
        V.copy().multAdd(before, C2);
        assertClose("argument on the output", C2, C1);
        CountingComplexMatrixD D1 = counting(6, 5, SEED + 31);
        ComplexMatrixD D2 = random(6, 5, SEED + 31);
        ComplexMatrixD beforeD = Matrices.view(D2, 0, 0, 3, 4).copy();
        V.mult(Matrices.view(D1, 0, 0, 3, 4), D1);
        assertEquals("argument on the zeroed output: copies", 1, D1.copies);
        V.copy().mult(beforeD, D2);
        assertClose("argument on the zeroed output", D2, D1);
    }

    @Test
    public void testTimesManyStillCopiesOnce() {
        for (boolean anchored : new boolean[] { false, true }) {
            CountingComplexMatrixD P = counting(10, 9, SEED + 32);
            ComplexMatrixD V = anchored ? Matrices.view(P, 0, 0, 5, 4) : Matrices.view(P, 2, 1, 7, 5);
            ComplexMatrixD B = random(5, 4, SEED + 33);
            ComplexMatrixD C = random(4, 3, SEED + 34);
            ComplexMatrixD D = random(3, 2, SEED + 35);
            ComplexMatrixD got = V.timesMany(B, C, D);
            assertEquals("copies", 1, P.copies);
            assertClose("timesMany", V.copy().timesMany(B, C, D), got);
        }
    }

    @Test
    public void testProductWithTheOutputBeingTheParentOfTheArgument() {
        for (boolean anchored : new boolean[] { false, true }) {
            int r0 = anchored ? 0 : 1;
            ComplexMatrixD A = random(6, 5, SEED + 11);
            ComplexMatrixD C1 = random(6, 4, SEED + 12);
            ComplexMatrixD C2 = random(6, 4, SEED + 12);
            ComplexMatrixD before = Matrices.view(C2, r0, 0, r0 + 4, 3).copy();
            assertSame(C1, A.multAdd(-1.0, 0.0, Matrices.view(C1, r0, 0, r0 + 4, 3), C1));
            A.multAdd(-1.0, 0.0, before, C2);
            assertClose("multAdd", C2, C1);

            ComplexMatrixD At = random(4, 5, SEED + 13);
            ComplexMatrixD D1 = random(5, 4, SEED + 14);
            ComplexMatrixD D2 = random(5, 4, SEED + 14);
            ComplexMatrixD beforeT = Matrices.view(D2, r0, 0, r0 + 3, 3).copy();
            assertSame(D1, At.conjTransAmultAdd(2.0, 1.0, Matrices.view(D1, r0, 0, r0 + 3, 3), D1));
            At.conjTransAmultAdd(2.0, 1.0, beforeT, D2);
            assertClose("conjTransAmultAdd", D2, D1);
        }
    }

    @Test
    public void testZeroedOutputIsNotReadThroughAView() {
        // A shape, C shape, B region within C
        OutputAlias[] cases = {
                new OutputAlias("mult", 6, 5, 6, 4, 1, 0, 5, 3, (A, B, C) -> A.mult(B, C)),
                new OutputAlias("mult at (0, 0)", 6, 5, 6, 4, 0, 0, 4, 3, (A, B, C) -> A.mult(B, C)),
                new OutputAlias("mult alpha", 6, 5, 6, 4, 1, 0, 5, 3, (A, B, C) -> A.mult(-2.0, 1.0, B, C)),
                new OutputAlias("conjTransAmult", 4, 5, 5, 4, 1, 0, 4, 3, (A, B, C) -> A.conjTransAmult(B, C)),
                new OutputAlias("conjTransAmult alpha", 4, 5, 5, 4, 1, 0, 4, 3,
                        (A, B, C) -> A.conjTransAmult(0.5, 0.5, B, C)),
                new OutputAlias("conjTransBmult", 7, 5, 7, 6, 1, 0, 6, 4, (A, B, C) -> A.conjTransBmult(B, C)),
                new OutputAlias("conjTransBmult alpha", 7, 5, 7, 6, 1, 0, 6, 4,
                        (A, B, C) -> A.conjTransBmult(3.0, 0.0, B, C)),
                new OutputAlias("conjTransABmult", 4, 6, 6, 5, 1, 1, 5, 4, (A, B, C) -> A.conjTransABmult(B, C)),
                new OutputAlias("conjTransABmult alpha", 4, 6, 6, 5, 1, 1, 5, 4,
                        (A, B, C) -> A.conjTransABmult(-1.0, -1.0, B, C)) };
        for (OutputAlias c : cases) {
            ComplexMatrixD A = random(c.rows, c.cols, SEED + 19);
            ComplexMatrixD C1 = random(c.outRows, c.outCols, SEED + 20);
            ComplexMatrixD C2 = random(c.outRows, c.outCols, SEED + 20);
            ComplexMatrixD before = Matrices.view(C2, c.r0, c.c0, c.r1, c.c1).copy();
            assertSame(c.name, C1, c.call.apply(A, Matrices.view(C1, c.r0, c.c0, c.r1, c.c1), C1));
            c.call.apply(A, before, C2);
            assertClose(c.name, C2, C1);
        }
    }

    @Test
    public void testEveryComplexMatrixDParameterIsClassified() {
        // implementations must not add public ComplexMatrixD parameters of their own
        for (Class<?> impl : new Class<?>[] { ComplexMatrixDBase.class, SimpleComplexMatrixD.class }) {
            for (Method m : impl.getMethods()) {
                if (hasComplexMatrixDParameter(m)) {
                    try {
                        ComplexMatrixD.class.getMethod(m.getName(), m.getParameterTypes());
                    } catch (NoSuchMethodException e) {
                        fail(impl.getSimpleName() + " declares " + key(impl.getSimpleName(), m, -1));
                    }
                }
            }
        }

        Set<String> found = new TreeSet<>();
        collect(found, "ComplexMatrixD", Arrays.asList(ComplexMatrixD.class.getMethods()));
        collect(found, "MatrixD", Arrays.asList(MatrixD.class.getMethods()));
        for (Class<?> c : new Class<?>[] { Matrices.class, Statistics.class }) {
            List<Executable> members = new ArrayList<>(Arrays.asList(c.getDeclaredMethods()));
            members.addAll(Arrays.asList(c.getConstructors()));
            collect(found, c.getSimpleName(), members);
        }

        Param[] table = params();
        Set<String> listed = new TreeSet<>();
        int writes = 0;
        for (Param p : table) {
            assertEquals("duplicate " + p.key, true, listed.add(p.key));
            writes += p.kind == Kind.WRITE ? 1 : 0;
        }
        Set<String> missing = new TreeSet<>(found);
        missing.removeAll(listed);
        Set<String> unknown = new TreeSet<>(listed);
        unknown.removeAll(found);
        assertEquals("not classified", "[]", missing.toString());
        assertEquals("not a parameter", "[]", unknown.toString());
        assertEquals("parameters", 96, table.length);
        assertEquals("writes", 27, writes);

        ComplexMatrixD P = random(60, 40, SEED + 18).addInplace(30.0, 0.0,
                Matrices.identityComplexD(60).selectConsecutiveColumns(0, 39));
        double[] parent = P.getArrayUnsafe().clone();
        for (Param p : table) {
            for (int off : new int[] { 4, 0 }) {
                String what = p.key + (off == 0 ? " at (0, 0)" : "");
                ComplexMatrixD V = Matrices.view(P, off, off, off + p.rows - 1, off + p.cols - 1);
                if (p.kind == Kind.WRITE) {
                    Ctx ctx = new Ctx(V);
                    try {
                        p.call.apply(ctx);
                        fail(what + ": no exception");
                    } catch (UnsupportedOperationException expected) {
                        ctx.assertUntouched(what);
                    }
                } else {
                    Object got = p.call.apply(new Ctx(V));
                    Object want = p.call.apply(new Ctx(V.copy()));
                    if (p.kind == Kind.BITS) {
                        assertResult(what, want, got);
                    } else {
                        assertClose(what, (ComplexMatrixD) want, (ComplexMatrixD) got);
                    }
                }
                assertBitsArray(what + ": parent", parent, P.getArrayUnsafe());
            }
        }
    }

    // every (member, ComplexMatrixD parameter) pair; x is the view or its copy
    static Param[] params() {
        String B = "(ComplexMatrixD,ComplexMatrixD)#";
        String aB = "(double,double,ComplexMatrixD,ComplexMatrixD)#";
        String Z = "ComplexMatrixD.";
        return new Param[] {
                // ComplexMatrixD: products read B and write C
                close(Z + "mult" + B + 0, 5, 4, c -> c.t(6, 5).mult(c.x, c.t(6, 4))),
                close(Z + "mult" + aB + 2, 5, 4, c -> c.t(6, 5).mult(0.5, 0.5, c.x, c.t(6, 4))),
                close(Z + "multAdd" + B + 0, 5, 4, c -> c.t(6, 5).multAdd(c.x, c.t(6, 4))),
                close(Z + "multAdd" + aB + 2, 5, 4, c -> c.t(6, 5).multAdd(2.0, 0.0, c.x, c.t(6, 4))),
                close(Z + "conjTransAmult" + B + 0, 6, 4, c -> c.t(6, 5).conjTransAmult(c.x, c.t(5, 4))),
                close(Z + "conjTransAmult" + aB + 2, 6, 4,
                        c -> c.t(6, 5).conjTransAmult(0.5, 1.0, c.x, c.t(5, 4))),
                close(Z + "conjTransAmultAdd" + B + 0, 6, 4, c -> c.t(6, 5).conjTransAmultAdd(c.x, c.t(5, 4))),
                close(Z + "conjTransAmultAdd" + aB + 2, 6, 4,
                        c -> c.t(6, 5).conjTransAmultAdd(2.0, -1.0, c.x, c.t(5, 4))),
                close(Z + "conjTransBmult" + B + 0, 4, 5, c -> c.t(6, 5).conjTransBmult(c.x, c.t(6, 4))),
                close(Z + "conjTransBmult" + aB + 2, 4, 5,
                        c -> c.t(6, 5).conjTransBmult(0.5, 0.0, c.x, c.t(6, 4))),
                close(Z + "conjTransBmultAdd" + B + 0, 4, 5, c -> c.t(6, 5).conjTransBmultAdd(c.x, c.t(6, 4))),
                close(Z + "conjTransBmultAdd" + aB + 2, 4, 5,
                        c -> c.t(6, 5).conjTransBmultAdd(2.0, 2.0, c.x, c.t(6, 4))),
                close(Z + "conjTransABmult" + B + 0, 4, 6, c -> c.t(6, 5).conjTransABmult(c.x, c.t(5, 4))),
                close(Z + "conjTransABmult" + aB + 2, 4, 6,
                        c -> c.t(6, 5).conjTransABmult(0.5, -0.5, c.x, c.t(5, 4))),
                close(Z + "conjTransABmultAdd" + B + 0, 4, 6, c -> c.t(6, 5).conjTransABmultAdd(c.x, c.t(5, 4))),
                close(Z + "conjTransABmultAdd" + aB + 2, 4, 6,
                        c -> c.t(6, 5).conjTransABmultAdd(2.0, 0.0, c.x, c.t(5, 4))),
                write(Z + "mult" + B + 1, 6, 4, c -> c.t(6, 5).mult(c.t(5, 4), c.x)),
                write(Z + "mult" + aB + 3, 6, 4, c -> c.t(6, 5).mult(0.5, 0.5, c.t(5, 4), c.x)),
                write(Z + "multAdd" + B + 1, 6, 4, c -> c.t(6, 5).multAdd(c.t(5, 4), c.x)),
                write(Z + "multAdd" + aB + 3, 6, 4, c -> c.t(6, 5).multAdd(2.0, 0.0, c.t(5, 4), c.x)),
                write(Z + "conjTransAmult" + B + 1, 5, 4, c -> c.t(6, 5).conjTransAmult(c.t(6, 4), c.x)),
                write(Z + "conjTransAmult" + aB + 3, 5, 4,
                        c -> c.t(6, 5).conjTransAmult(0.5, 1.0, c.t(6, 4), c.x)),
                write(Z + "conjTransAmultAdd" + B + 1, 5, 4, c -> c.t(6, 5).conjTransAmultAdd(c.t(6, 4), c.x)),
                write(Z + "conjTransAmultAdd" + aB + 3, 5, 4,
                        c -> c.t(6, 5).conjTransAmultAdd(2.0, -1.0, c.t(6, 4), c.x)),
                write(Z + "conjTransBmult" + B + 1, 6, 4, c -> c.t(6, 5).conjTransBmult(c.t(4, 5), c.x)),
                write(Z + "conjTransBmult" + aB + 3, 6, 4,
                        c -> c.t(6, 5).conjTransBmult(0.5, 0.0, c.t(4, 5), c.x)),
                write(Z + "conjTransBmultAdd" + B + 1, 6, 4, c -> c.t(6, 5).conjTransBmultAdd(c.t(4, 5), c.x)),
                write(Z + "conjTransBmultAdd" + aB + 3, 6, 4,
                        c -> c.t(6, 5).conjTransBmultAdd(2.0, 2.0, c.t(4, 5), c.x)),
                write(Z + "conjTransABmult" + B + 1, 5, 4, c -> c.t(6, 5).conjTransABmult(c.t(4, 6), c.x)),
                write(Z + "conjTransABmult" + aB + 3, 5, 4,
                        c -> c.t(6, 5).conjTransABmult(0.5, -0.5, c.t(4, 6), c.x)),
                write(Z + "conjTransABmultAdd" + B + 1, 5, 4, c -> c.t(6, 5).conjTransABmultAdd(c.t(4, 6), c.x)),
                write(Z + "conjTransABmultAdd" + aB + 3, 5, 4,
                        c -> c.t(6, 5).conjTransABmultAdd(2.0, 0.0, c.t(4, 6), c.x)),
                // ComplexMatrixD: element-wise reads, in place on the receiver
                bits(Z + "add" + B + 0, 6, 5, c -> c.t(6, 5).add(c.x, c.t(6, 5))),
                bits(Z + "add" + aB + 2, 6, 5, c -> c.t(6, 5).add(-1.5, 0.5, c.x, c.t(6, 5))),
                write(Z + "add" + B + 1, 6, 5, c -> c.t(6, 5).add(c.t(6, 5), c.x)),
                write(Z + "add" + aB + 3, 6, 5, c -> c.t(6, 5).add(-1.5, 0.5, c.t(6, 5), c.x)),
                bits(Z + "addInplace(ComplexMatrixD)#0", 6, 5, c -> c.t(6, 5).addInplace(c.x)),
                bits(Z + "addInplace(double,double,ComplexMatrixD)#2", 6, 5,
                        c -> c.t(6, 5).addInplace(0.5, -0.5, c.x)),
                bits(Z + "setSubmatrixInplace(int,int,ComplexMatrixD,int,int,int,int)#2", 10, 4,
                        c -> c.t(12, 6).setSubmatrixInplace(1, 1, c.x, 0, 0, 9, 3)),
                bits(Z + "setInplaceUpperTrapezoidal(ComplexMatrixD)#0", 7, 6,
                        c -> c.t(6, 5).setInplaceUpperTrapezoidal(c.x)),
                bits(Z + "setInplaceLowerTrapezoidal(ComplexMatrixD)#0", 7, 6,
                        c -> c.t(6, 5).setInplaceLowerTrapezoidal(c.x)),
                bits(Z + "setColumnInplace(int,ComplexMatrixD)#1", 6, 1, c -> c.t(6, 5).setColumnInplace(2, c.x)),
                bits(Z + "setInplace(ComplexMatrixD)#0", 6, 5, c -> c.t(6, 5).setInplace(c.x)),
                bits(Z + "setInplace(double,double,ComplexMatrixD)#2", 6, 5,
                        c -> c.t(6, 5).setInplace(3.0, 1.0, c.x)),
                bits(Z + "hadamard" + B + 0, 6, 5, c -> c.t(6, 5).hadamard(c.x, c.t(6, 5))),
                write(Z + "hadamard" + B + 1, 6, 5, c -> c.t(6, 5).hadamard(c.t(6, 5), c.x)),
                close(Z + "solve" + B + 0, 5, 2, c -> c.sq(5).solve(c.x, c.t(5, 2))),
                write(Z + "solve" + B + 1, 5, 2, c -> c.sq(5).solve(c.t(5, 2), c.x)),
                write(Z + "trans(ComplexMatrixD)#0", 5, 6, c -> c.t(6, 5).trans(c.x)),
                write(Z + "conjTrans(ComplexMatrixD)#0", 5, 6, c -> c.t(6, 5).conjTrans(c.x)),
                write(Z + "scale(double,double,ComplexMatrixD)#2", 6, 5, c -> c.t(6, 5).scale(2.0, 1.0, c.x)),
                write(Z + "inv(ComplexMatrixD)#0", 5, 5, c -> c.sq(5).inv(c.x)),
                write(Z + "submatrix(int,int,int,int,ComplexMatrixD,int,int)#4", 3, 3,
                        c -> c.t(6, 5).submatrix(0, 0, 2, 2, c.x, 0, 0)),
                // ComplexMatrixD: reads that allocate their result
                bits(Z + "plus(ComplexMatrixD)#0", 6, 5, c -> c.t(6, 5).plus(c.x)),
                bits(Z + "minus(ComplexMatrixD)#0", 6, 5, c -> c.t(6, 5).minus(c.x)),
                bits(Z + "hadamard(ComplexMatrixD)#0", 6, 5, c -> c.t(6, 5).hadamard(c.x)),
                bits(Z + "appendColumn(ComplexMatrixD)#0", 6, 1, c -> c.t(6, 5).appendColumn(c.x)),
                bits(Z + "appendMatrix(ComplexMatrixD)#0", 6, 3, c -> c.t(6, 5).appendMatrix(c.x)),
                close(Z + "times(ComplexMatrixD)#0", 5, 4, c -> c.t(6, 5).times(c.x)),
                close(Z + "timesTimes" + B + 0, 5, 4, c -> c.t(6, 5).timesTimes(c.x, c.t(4, 3))),
                close(Z + "timesTimes" + B + 1, 4, 3, c -> c.t(6, 5).timesTimes(c.t(5, 4), c.x)),
                close(Z + "timesMany(ComplexMatrixD,ComplexMatrixD[])#0", 5, 4,
                        c -> c.t(6, 5).timesMany(c.x, c.t(4, 3), c.t(3, 2))),
                close(Z + "timesMany(ComplexMatrixD,ComplexMatrixD[])#1", 4, 3,
                        c -> c.t(6, 5).timesMany(c.t(5, 4), c.x, c.t(3, 2))),
                close(Z + "timesPlus" + B + 0, 5, 4, c -> c.t(6, 5).timesPlus(c.x, c.t(6, 4))),
                close(Z + "timesPlus" + B + 1, 6, 4, c -> c.t(6, 5).timesPlus(c.t(5, 4), c.x)),
                close(Z + "timesMinus" + B + 0, 5, 4, c -> c.t(6, 5).timesMinus(c.x, c.t(6, 4))),
                close(Z + "timesMinus" + B + 1, 6, 4, c -> c.t(6, 5).timesMinus(c.t(5, 4), c.x)),
                close(Z + "timesConjugateTransposed(ComplexMatrixD)#0", 3, 5,
                        c -> c.t(6, 5).timesConjugateTransposed(c.x)),
                close(Z + "conjugateTransposedTimes(ComplexMatrixD)#0", 6, 3,
                        c -> c.t(6, 5).conjugateTransposedTimes(c.x)),
                close(Z + "mldivide(ComplexMatrixD)#0", 5, 2, c -> c.sq(5).mldivide(c.x)),
                close(Z + "mrdivide(ComplexMatrixD)#0", 5, 5, c -> c.t(4, 5).mrdivide(c.x)),
                close("MatrixD.times(ComplexMatrixD)#0", 5, 4,
                        c -> Matrices.randomUniformD(6, 5, SEED).times(c.x)),
                // Matrices
                bits("Matrices.view(ComplexMatrixD,int,int,int,int)#0", 6, 5,
                        c -> Matrices.view(c.x, 1, 1, 4, 3).copy()),
                bits("Matrices.embed(int,int,ComplexMatrixD)#2", 6, 5, c -> Matrices.embed(8, 7, c.x)),
                bits("Matrices.sameDimComplexD(ComplexMatrixD)#0", 6, 5, c -> Matrices.sameDimComplexD(c.x)),
                bits("Matrices.serializeComplexD(ComplexMatrixD,Path)#0", 6, 5, c -> serializeToFile(c.x)),
                bits("Matrices.serializeComplexD(ComplexMatrixD,OutputStream)#0", 6, 5, c -> serialize(c.x)),
                bits("Matrices.convert(ComplexMatrixD)#0", 6, 5, c -> Matrices.convert(c.x)),
                bits("Matrices.convertToReal(ComplexMatrixD)#0", 6, 5, c -> Matrices.convertToReal(c.x)),
                bits("Matrices.round(ComplexMatrixD,int)#0", 6, 5, c -> Matrices.round(c.x, 4)),
                bits("Matrices.sumRows(ComplexMatrixD)#0", 6, 5, c -> Matrices.sumRows(c.x)),
                bits("Matrices.sumColumns(ComplexMatrixD)#0", 6, 5, c -> Matrices.sumColumns(c.x)),
                bits("Matrices.distance(ComplexMatrixD,ComplexMatrixD)#0", 6, 5,
                        c -> Matrices.distance(c.x, c.t(6, 5))),
                bits("Matrices.distance(ComplexMatrixD,ComplexMatrixD)#1", 6, 5,
                        c -> Matrices.distance(c.t(6, 5), c.x)),
                bits("Matrices.approxEqual(ComplexMatrixD,ComplexMatrixD)#0", 6, 5,
                        c -> Matrices.approxEqual(c.x, c.x.copy())),
                bits("Matrices.approxEqual(ComplexMatrixD,ComplexMatrixD)#1", 6, 5,
                        c -> Matrices.approxEqual(c.t(6, 5), c.x)),
                bits("Matrices.approxEqual(ComplexMatrixD,ComplexMatrixD,double)#0", 6, 5,
                        c -> Matrices.approxEqual(c.x, c.x.copy(), 1.0e-3)),
                bits("Matrices.approxEqual(ComplexMatrixD,ComplexMatrixD,double)#1", 6, 5,
                        c -> Matrices.approxEqual(c.t(6, 5), c.x, 1.0)),
                bits("Matrices.approxEqual(ComplexMatrixD,ComplexMatrixD,double,double)#0", 6, 5,
                        c -> Matrices.approxEqual(c.x, c.t(6, 5), 0.0, 1.0)),
                bits("Matrices.approxEqual(ComplexMatrixD,ComplexMatrixD,double,double)#1", 6, 5,
                        c -> Matrices.approxEqual(c.t(6, 5), c.x, 0.0, 0.1)),
                bits("Matrices.numericalRank(ComplexMatrixD)#0", 6, 5, c -> Matrices.numericalRank(c.x)),
                bits("Matrices.numericalRank(ComplexMatrixD,double)#0", 6, 5,
                        c -> Matrices.numericalRank(c.x, 1.0)),
                // Statistics
                bits("Statistics.centerColumns(ComplexMatrixD)#0", 6, 5, c -> Statistics.centerColumns(c.x)),
                bits("Statistics.zscoreColumns(ComplexMatrixD)#0", 6, 5, c -> Statistics.zscoreColumns(c.x)),
                write("Statistics.centerColumnsInplace(ComplexMatrixD)#0", 6, 5,
                        c -> Statistics.centerColumnsInplace(c.x)),
                write("Statistics.zscoreColumnsInplace(ComplexMatrixD)#0", 6, 5,
                        c -> Statistics.zscoreColumnsInplace(c.x)) };
    }

    static Prod[] products() {
        return new Prod[] {
                new Prod("mult", s(5, 4), (T, x) -> T.mult(x[0], Matrices.createComplexD(6, 4))),
                new Prod("mult alpha", s(5, 4), (T, x) -> T.mult(-0.5, 0.25, x[0], Matrices.createComplexD(6, 4))),
                new Prod("multAdd", s(5, 4), (T, x) -> T.multAdd(x[0], filled(6, 4))),
                new Prod("multAdd alpha", s(5, 4), (T, x) -> T.multAdd(1.5, -1.0, x[0], filled(6, 4))),
                new Prod("conjTransABmult", s(4, 6), (T, x) -> T.conjTransABmult(x[0], Matrices.createComplexD(5, 4))),
                new Prod("conjTransABmult alpha", s(4, 6),
                        (T, x) -> T.conjTransABmult(2.0, 1.0, x[0], Matrices.createComplexD(5, 4))),
                new Prod("conjTransABmultAdd", s(4, 6), (T, x) -> T.conjTransABmultAdd(x[0], filled(5, 4))),
                new Prod("conjTransABmultAdd alpha", s(4, 6),
                        (T, x) -> T.conjTransABmultAdd(-1.0, 0.0, x[0], filled(5, 4))),
                new Prod("conjTransAmult", s(6, 4), (T, x) -> T.conjTransAmult(x[0], Matrices.createComplexD(5, 4))),
                new Prod("conjTransAmult alpha", s(6, 4),
                        (T, x) -> T.conjTransAmult(3.0, -3.0, x[0], Matrices.createComplexD(5, 4))),
                new Prod("conjTransAmultAdd", s(6, 4), (T, x) -> T.conjTransAmultAdd(x[0], filled(5, 4))),
                new Prod("conjTransAmultAdd alpha", s(6, 4),
                        (T, x) -> T.conjTransAmultAdd(0.25, 0.5, x[0], filled(5, 4))),
                new Prod("conjTransBmult", s(4, 5), (T, x) -> T.conjTransBmult(x[0], Matrices.createComplexD(6, 4))),
                new Prod("conjTransBmult alpha", s(4, 5),
                        (T, x) -> T.conjTransBmult(-2.0, 0.5, x[0], Matrices.createComplexD(6, 4))),
                new Prod("conjTransBmultAdd", s(4, 5), (T, x) -> T.conjTransBmultAdd(x[0], filled(6, 4))),
                new Prod("conjTransBmultAdd alpha", s(4, 5),
                        (T, x) -> T.conjTransBmultAdd(0.5, 0.0, x[0], filled(6, 4))),
                new Prod("times", s(5, 4), (T, x) -> T.times(x[0])),
                new Prod("timesTimes", s(5, 4, 4, 3), (T, x) -> T.timesTimes(x[0], x[1])),
                new Prod("timesMany", s(5, 4, 4, 3, 3, 2), (T, x) -> T.timesMany(x[0], x[1], x[2])),
                new Prod("timesConjugateTransposed", s(3, 5), (T, x) -> T.timesConjugateTransposed(x[0])),
                new Prod("conjugateTransposedTimes", s(6, 3), (T, x) -> T.conjugateTransposedTimes(x[0])),
                new Prod("timesPlus", s(5, 4, 6, 4), (T, x) -> T.timesPlus(x[0], x[1])),
                new Prod("timesMinus", s(5, 4, 6, 4), (T, x) -> T.timesMinus(x[0], x[1])) };
    }

    // the i-th argument of a product case, at (0, 0) or at an offset that differs per argument
    static ComplexMatrixD productView(ComplexMatrixD P, Prod c, int i, boolean anchored) {
        int r0 = anchored ? 0 : 1 + 3 * i;
        int c0 = anchored ? 0 : 2 + 2 * i;
        return Matrices.view(P, r0, c0, r0 + c.shapes[2 * i] - 1, c0 + c.shapes[2 * i + 1] - 1);
    }

    // ---------------------------------------------------------------- cases

    static final class ArgCase {
        final String name;
        final int rows;
        final int cols;
        final int argRows;
        final int argCols;
        final BiFunction<ComplexMatrixD, ComplexMatrixD, Object> call;

        ArgCase(String name, int rows, int cols, int argRows, int argCols,
                BiFunction<ComplexMatrixD, ComplexMatrixD, Object> call) {
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
        final BiConsumer<ComplexMatrixD, ComplexMatrixD> call;

        Overlap(String name, int rows, int cols, int r0, int c0, int r1, int c1,
                BiConsumer<ComplexMatrixD, ComplexMatrixD> call) {
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
        final BiFunction<ComplexMatrixD, ComplexMatrixD, Object> call;

        Stat(String name, boolean exact, int[] a, int[] b, BiFunction<ComplexMatrixD, ComplexMatrixD, Object> call) {
            this.name = name;
            this.exact = exact;
            this.a = a;
            this.b = b;
            this.call = call;
        }
    }

    // counts the block copies a view of this matrix makes
    static final class CountingComplexMatrixD extends SimpleComplexMatrixD {
        int copies;

        CountingComplexMatrixD(int rows, int cols) {
            super(rows, cols);
        }

        @Override
        public ComplexMatrixD submatrix(int r0, int c0, int r1, int c1, ComplexMatrixD B, int rb, int cb) {
            ++copies;
            return super.submatrix(r0, c0, r1, c1, B, rb, cb);
        }
    }

    static CountingComplexMatrixD counting(int rows, int cols, long seed) {
        CountingComplexMatrixD m = new CountingComplexMatrixD(rows, cols);
        m.setInplace(random(rows, cols, seed));
        return m;
    }

    interface TriFunction {
        ComplexMatrixD apply(ComplexMatrixD A, ComplexMatrixD B, ComplexMatrixD C);
    }

    static final class ParentOut {
        final String name;
        final int rows;
        final int cols;
        final int argRows;
        final int argCols;
        final int outRows;
        final int outCols;
        final TriFunction call;

        ParentOut(String name, int rows, int cols, int argRows, int argCols, int outRows, int outCols,
                TriFunction call) {
            this.name = name;
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
        final BiFunction<ComplexMatrixD, ComplexMatrixD[], ComplexMatrixD> call;

        Prod(String name, int[] shapes, BiFunction<ComplexMatrixD, ComplexMatrixD[], ComplexMatrixD> call) {
            this.name = name;
            this.shapes = shapes;
            this.call = call;
        }
    }

    enum Kind {
        BITS, CLOSE, WRITE
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

    static Param write(String key, int rows, int cols, Function<Ctx, Object> call) {
        return new Param(key, Kind.WRITE, rows, cols, call);
    }

    // the argument under test plus the deterministic matrices a call creates
    static final class Ctx {
        final ComplexMatrixD x;
        final List<ComplexMatrixD> created = new ArrayList<>();
        final List<Supplier<ComplexMatrixD>> fresh = new ArrayList<>();

        Ctx(ComplexMatrixD x) {
            this.x = x;
        }

        ComplexMatrixD t(int rows, int cols) {
            long seed = SEED * 31 + created.size();
            return track(() -> random(rows, cols, seed));
        }

        // a well-conditioned square matrix
        ComplexMatrixD sq(int n) {
            long seed = SEED * 37 + created.size();
            return track(() -> random(n, n, seed).addInplace(10.0 * n, 0.0, Matrices.identityComplexD(n)));
        }

        ComplexMatrixD track(Supplier<ComplexMatrixD> s) {
            ComplexMatrixD m = s.get();
            created.add(m);
            fresh.add(s);
            return m;
        }

        void assertUntouched(String what) {
            for (int i = 0; i < created.size(); ++i) {
                assertBitsArray(what + ": argument " + i, fresh.get(i).get().getArrayUnsafe(),
                        created.get(i).getArrayUnsafe());
            }
        }
    }

    // ---------------------------------------------------------------- helpers

    static boolean hasComplexMatrixDParameter(Executable e) {
        for (Class<?> t : e.getParameterTypes()) {
            if (t == ComplexMatrixD.class || t == ComplexMatrixD[].class) {
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
                if (t == ComplexMatrixD.class || t == ComplexMatrixD[].class) {
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
    static ComplexMatrixD filled(int rows, int cols) {
        return random(rows, cols, SEED - 1);
    }

    // the result, or the class of the runtime exception thrown
    static Object outcome(Supplier<Object> call) {
        try {
            return call.get();
        } catch (RuntimeException e) {
            return e.getClass();
        }
    }

    static byte[] serialize(ComplexMatrixD m) {
        try {
            ByteArrayOutputStream os = new ByteArrayOutputStream();
            long n = Matrices.serializeComplexD(m, os);
            assertEquals("bytes written", os.size(), n);
            return os.toByteArray();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static byte[] serializeToFile(ComplexMatrixD m) {
        Path file = null;
        try {
            file = Files.createTempFile("jamu-view", ".bin");
            Matrices.serializeComplexD(m, file);
            return Files.readAllBytes(file);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } finally {
            if (file != null) {
                file.toFile().delete();
            }
        }
    }

    // values in [1, 2) so that no entry is zero
    static ComplexMatrixD random(int rows, int cols, long seed) {
        return Matrices.randomUniformComplexD(rows, cols, 1.0, 2.0, seed);
    }

    static void assertResult(String what, Object want, Object got) {
        if (want instanceof ComplexMatrixD) {
            if (!(got instanceof ComplexMatrixD)) {
                fail(what + ": got " + got);
            }
            ComplexMatrixD w = (ComplexMatrixD) want;
            ComplexMatrixD g = (ComplexMatrixD) got;
            assertEquals(what + ": rows", w.numRows(), g.numRows());
            assertEquals(what + ": cols", w.numColumns(), g.numColumns());
            assertBitsArray(what, w.getArrayUnsafe(), g.getArrayUnsafe());
        } else if (want instanceof ComplexMatrixF) {
            ComplexMatrixF w = (ComplexMatrixF) want;
            ComplexMatrixF g = (ComplexMatrixF) got;
            assertEquals(what + ": rows", w.numRows(), g.numRows());
            assertEquals(what + ": cols", w.numColumns(), g.numColumns());
            float[] fw = w.getArrayUnsafe();
            float[] fg = g.getArrayUnsafe();
            assertEquals(what + ": length", fw.length, fg.length);
            for (int i = 0; i < fw.length; ++i) {
                assertEquals(what + " [" + i + "]", Float.floatToRawIntBits(fw[i]), Float.floatToRawIntBits(fg[i]));
            }
        } else if (want instanceof MatrixD) {
            MatrixD w = (MatrixD) want;
            MatrixD g = (MatrixD) got;
            assertEquals(what + ": rows", w.numRows(), g.numRows());
            assertEquals(what + ": cols", w.numColumns(), g.numColumns());
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
