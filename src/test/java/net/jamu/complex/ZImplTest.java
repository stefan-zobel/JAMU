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
package net.jamu.complex;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Tests for the mutable complex implementations.
 */
public final class ZImplTest {

    /** the first exponent, the last one and the step of the test ensemble */
    private static final int MIN_EXP_D = -300;
    private static final int MAX_EXP_D = 300;
    private static final int MIN_EXP_F = -30;
    private static final int MAX_EXP_F = 30;
    private static final int ANGLES = 8;

    private static double angle(int k) {
        // 0.3 keeps the components off the axes
        return k * Math.PI / 4.0 + 0.3;
    }

    private static double relative(Zd got, Zd want) {
        return new ZdImpl(got.re() - want.re(), got.im() - want.im()).abs() / want.abs();
    }

    @Test
    public void testInvAgreesWithDivDouble() {
        for (int e = MIN_EXP_D; e <= MAX_EXP_D; ++e) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                double re = r * Math.cos(angle(k));
                double im = r * Math.sin(angle(k));
                Zd got = new ZdImpl(re, im).inv();
                Zd want = new ZdImpl(1.0, 0.0).div(new ZdImpl(re, im));
                assertEquals("re at 1e" + e, want.re(), got.re(), 0.0);
                assertEquals("im at 1e" + e, want.im(), got.im(), 0.0);
            }
        }
    }

    @Test
    public void testInvAgreesWithDivFloat() {
        for (int e = MIN_EXP_F; e <= MAX_EXP_F; ++e) {
            float r = (float) Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                float re = (float) (r * Math.cos(angle(k)));
                float im = (float) (r * Math.sin(angle(k)));
                Zf got = new ZfImpl(re, im).inv();
                Zf want = new ZfImpl(1.0f, 0.0f).div(new ZfImpl(re, im));
                assertEquals("re at 1e" + e, want.re(), got.re(), 0.0f);
                assertEquals("im at 1e" + e, want.im(), got.im(), 0.0f);
            }
        }
    }

    @Test
    public void testInvRoundTrip() {
        for (int e = MIN_EXP_D; e <= MAX_EXP_D; ++e) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                Zd z = new ZdImpl(r * Math.cos(angle(k)), r * Math.sin(angle(k)));
                assertTrue("1e" + e, relative(z.copy().inv().inv(), z) <= 1.0e-15);
            }
        }
    }

    @Test
    public void testTimesInverseIsOne() {
        for (int e = MIN_EXP_D; e <= MAX_EXP_D; ++e) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                Zd z = new ZdImpl(r * Math.cos(angle(k)), r * Math.sin(angle(k)));
                assertTrue("1e" + e, relative(z.copy().mul(z.copy().inv()), Zd.One()) <= 1.0e-15);
            }
        }
    }

    @Test
    public void testInvWhereTheSquaredModulusWouldOverflow() {
        Zf f = new ZfImpl(1.0e20f, 1.0e20f).inv();
        assertEquals("float re", 5.0e-21f, f.re(), 1.0e-26f);
        assertEquals("float im", -5.0e-21f, f.im(), 1.0e-26f);
        Zd d = new ZdImpl(1.0e200, 1.0e200).inv();
        assertEquals("double re", 5.0e-201, d.re(), 1.0e-215);
        assertEquals("double im", -5.0e-201, d.im(), 1.0e-215);
        Zd u = new ZdImpl(1.0e-200, 1.0e-200).inv();
        assertEquals("double underflow re", 5.0e+199, u.re(), 1.0e+185);
        assertEquals("double underflow im", -5.0e+199, u.im(), 1.0e+185);
    }

    /** e^(re + im i) through the finite branch only, scaled back up */
    private static double[] expReference(double re, double im) {
        Zd w = new ZdImpl(re - 300.0, im).exp();
        double f = Math.exp(300.0);
        return new double[] { w.re() * f, w.im() * f };
    }

    private static boolean usable(double x) {
        return !Double.isInfinite(x) && !Double.isNaN(x) && x != 0.0;
    }

    @Test
    public void testExpOverflowKeepsAnExactZero() {
        Zd d = new ZdImpl(1000.0, 0.0).exp();
        assertEquals(Double.POSITIVE_INFINITY, d.re(), 0.0);
        assertEquals("inf times an exact zero must stay zero", 0.0, d.im(), 0.0);
        Zf f = new ZfImpl(1000.0f, 0.0f).exp();
        assertEquals(Float.POSITIVE_INFINITY, f.re(), 0.0f);
        assertEquals("inf times an exact zero must stay zero", 0.0f, f.im(), 0.0f);
        Zd u = new ZdImpl(-1000.0, 0.0).exp();
        assertEquals(0.0, u.re(), 0.0);
        assertEquals(0.0, u.im(), 0.0);
    }

    @Test
    public void testExpInTheOverflowBand() {
        // e^re overflows here, the product with cos or sin need not
        for (double re : new double[] { 709.9, 710.0, 712.0, 715.0, 720.0, 740.0 }) {
            for (double im : new double[] { 1.0, 1.5, Math.PI / 2.0 }) {
                Zd got = new ZdImpl(re, im).exp();
                double[] want = expReference(re, im);
                if (usable(want[0])) {
                    assertTrue("re at " + re + ", " + im, usable(got.re()));
                    assertTrue("re at " + re + ", " + im, Math.abs(got.re() - want[0]) <= 1.0e-14 * Math.abs(want[0]));
                }
                if (usable(want[1])) {
                    assertTrue("im at " + re + ", " + im, usable(got.im()));
                    assertTrue("im at " + re + ", " + im, Math.abs(got.im() - want[1]) <= 1.0e-14 * Math.abs(want[1]));
                }
            }
        }
    }

    @Test
    public void testExpDoesNotInventFiniteValues() {
        for (double re = 709.79; re <= 800.0; re += 0.37) {
            for (int k = 0; k < 40; ++k) {
                double im = k * 0.157 - 3.0;
                Zd got = new ZdImpl(re, im).exp();
                double[] want = expReference(re, im);
                if (Double.isInfinite(want[0])) {
                    assertTrue("re at " + re, Double.isInfinite(got.re()));
                }
                if (Double.isInfinite(want[1])) {
                    assertTrue("im at " + re, Double.isInfinite(got.im()));
                }
            }
        }
    }

    @Test
    public void testExpBelowOverflowIsUnchanged() {
        for (int r = -700; r <= 700; r += 7) {
            for (int k = 0; k < 12; ++k) {
                double im = k * 0.7 - 4.0;
                Zd got = new ZdImpl(r, im).exp();
                double h = Math.exp(r);
                assertEquals("re at " + r, Double.doubleToLongBits(h * Math.cos(im)),
                        Double.doubleToLongBits(got.re()));
                assertEquals("im at " + r, Double.doubleToLongBits(h * Math.sin(im)),
                        Double.doubleToLongBits(got.im()));
            }
        }
    }

    @Test
    public void testPowOfALargeRealStaysReal() {
        // pow goes through ln().scale().exp(), so it inherits the exp fix
        Zd p = new ZdImpl(1.0e200, 0.0).pow(2.0);
        assertEquals(Double.POSITIVE_INFINITY, p.re(), 0.0);
        assertEquals(0.0, p.im(), 0.0);
    }

    private static void assertZ(String what, double wantRe, double wantIm, Zd got) {
        assertEquals(what + " re", wantRe, got.re(), 0.0);
        assertEquals(what + " im", wantIm, got.im(), 0.0);
    }

    @Test
    public void testPowOfZero() {
        double inf = Double.POSITIVE_INFINITY;
        assertZ("0^2", 0.0, 0.0, new ZdImpl(0.0, 0.0).pow(2.0));
        assertZ("0^0.5", 0.0, 0.0, new ZdImpl(0.0, 0.0).pow(0.5));
        assertZ("0^0", 1.0, 0.0, new ZdImpl(0.0, 0.0).pow(0.0));
        assertZ("0^-2", inf, inf, new ZdImpl(0.0, 0.0).pow(-2.0));
        assertZ("0^NaN", Double.NaN, Double.NaN, new ZdImpl(0.0, 0.0).pow(Double.NaN));
        Zf f = new ZfImpl(0.0f, 0.0f).pow(2.0f);
        assertEquals("float 0^2 re", 0.0f, f.re(), 0.0f);
        assertEquals("float 0^2 im", 0.0f, f.im(), 0.0f);
        Zf g = new ZfImpl(0.0f, 0.0f).pow(0.0f);
        assertEquals("float 0^0 re", 1.0f, g.re(), 0.0f);
        assertEquals("float 0^0 im", 0.0f, g.im(), 0.0f);
    }

    @Test
    public void testPowOfInfinity() {
        double inf = Double.POSITIVE_INFINITY;
        assertZ("inf^2", inf, inf, new ZdImpl(inf, 0.0).pow(2.0));
        assertZ("inf^0", 1.0, 0.0, new ZdImpl(inf, 0.0).pow(0.0));
        assertZ("inf^-2", 0.0, 0.0, new ZdImpl(inf, 0.0).pow(-2.0));
        Zf f = new ZfImpl(Float.POSITIVE_INFINITY, 0.0f).pow(-2.0f);
        assertEquals("float inf^-2 re", 0.0f, f.re(), 0.0f);
        assertEquals("float inf^-2 im", 0.0f, f.im(), 0.0f);
    }

    @Test
    public void testPowOfZeroWithAComplexExponent() {
        double inf = Double.POSITIVE_INFINITY;
        assertZ("0^(2+0i)", 0.0, 0.0, new ZdImpl(0.0, 0.0).pow(new ZdImpl(2.0, 0.0)));
        assertZ("0^(0+0i)", 1.0, 0.0, new ZdImpl(0.0, 0.0).pow(new ZdImpl(0.0, 0.0)));
        assertZ("0^(0+1i)", Double.NaN, Double.NaN, new ZdImpl(0.0, 0.0).pow(new ZdImpl(0.0, 1.0)));
        assertZ("0^(-2+3i)", inf, inf, new ZdImpl(0.0, 0.0).pow(new ZdImpl(-2.0, 3.0)));
    }

    @Test
    public void testPowWithANonFiniteExponent() {
        // only a real exponent is defined for a degenerate base
        double inf = Double.POSITIVE_INFINITY;
        assertZ("0^(2+inf i)", Double.NaN, Double.NaN, new ZdImpl(0.0, 0.0).pow(new ZdImpl(2.0, inf)));
        assertZ("inf^(1+inf i)", Double.NaN, Double.NaN, new ZdImpl(inf, 0.0).pow(new ZdImpl(1.0, inf)));
        assertZ("0^(2+NaN i)", Double.NaN, Double.NaN, new ZdImpl(0.0, 0.0).pow(new ZdImpl(2.0, Double.NaN)));
        Zf f = new ZfImpl(0.0f, 0.0f).pow(new ZfImpl(2.0f, Float.POSITIVE_INFINITY));
        assertEquals("float 0^(2+inf i) re", Float.NaN, f.re(), 0.0f);
        assertEquals("float 0^(2+inf i) im", Float.NaN, f.im(), 0.0f);
        // a real exponent stays real, so both overloads agree
        assertZ("0^(-inf+0i)", inf, inf, new ZdImpl(0.0, 0.0).pow(new ZdImpl(Double.NEGATIVE_INFINITY, 0.0)));
        assertZ("0^-inf", inf, inf, new ZdImpl(0.0, 0.0).pow(Double.NEGATIVE_INFINITY));
    }

    @Test
    public void testPowWithAnInfiniteExponent() {
        // the modulus of the base decides, as |x| does in Math.pow
        double inf = Double.POSITIVE_INFINITY;
        double neg = Double.NEGATIVE_INFINITY;
        assertZ("2^inf", inf, inf, new ZdImpl(2.0, 0.0).pow(inf));
        assertZ("2^-inf", 0.0, 0.0, new ZdImpl(2.0, 0.0).pow(neg));
        assertZ("0.5^inf", 0.0, 0.0, new ZdImpl(0.5, 0.0).pow(inf));
        assertZ("0.5^-inf", inf, inf, new ZdImpl(0.5, 0.0).pow(neg));
        assertZ("(3,4)^inf", inf, inf, new ZdImpl(3.0, 4.0).pow(inf));
        assertZ("(0.3,0.4)^inf", 0.0, 0.0, new ZdImpl(0.3, 0.4).pow(inf));
    }

    @Test
    public void testPowWithAnInfiniteExponentAtModulusOne() {
        double inf = Double.POSITIVE_INFINITY;
        assertZ("1^inf", Double.NaN, Double.NaN, new ZdImpl(1.0, 0.0).pow(inf));
        assertZ("i^inf", Double.NaN, Double.NaN, new ZdImpl(0.0, 1.0).pow(inf));
        assertZ("NaN base", Double.NaN, Double.NaN, new ZdImpl(Double.NaN, 1.0).pow(inf));
    }

    @Test
    public void testPowWithAnInfiniteExponentFarOutOfRange() {
        // the squared modulus would over- and underflow here, abs() does not
        double inf = Double.POSITIVE_INFINITY;
        assertZ("(1e200,1e200)^inf", inf, inf, new ZdImpl(1.0e200, 1.0e200).pow(inf));
        assertZ("(1e-200,1e-200)^inf", 0.0, 0.0, new ZdImpl(1.0e-200, 1.0e-200).pow(inf));
    }

    @Test
    public void testInfiniteExponentAgreesAcrossTheOverloads() {
        double inf = Double.POSITIVE_INFINITY;
        assertZ("2^(inf+0i)", inf, inf, new ZdImpl(2.0, 0.0).pow(new ZdImpl(inf, 0.0)));
        assertZ("2^(-inf+0i)", 0.0, 0.0, new ZdImpl(2.0, 0.0).pow(new ZdImpl(Double.NEGATIVE_INFINITY, 0.0)));
        assertZ("2^(1+inf i)", Double.NaN, Double.NaN, new ZdImpl(2.0, 0.0).pow(new ZdImpl(1.0, inf)));
        Zf f = new ZfImpl(2.0f, 0.0f).pow(Float.POSITIVE_INFINITY);
        assertEquals("float 2^inf re", Float.POSITIVE_INFINITY, f.re(), 0.0f);
        assertEquals("float 2^inf im", Float.POSITIVE_INFINITY, f.im(), 0.0f);
        Zf g = new ZfImpl(0.5f, 0.0f).pow(Float.POSITIVE_INFINITY);
        assertEquals("float 0.5^inf re", 0.0f, g.re(), 0.0f);
        assertEquals("float 0.5^inf im", 0.0f, g.im(), 0.0f);
        Zf h = new ZfImpl(1.0f, 0.0f).pow(new ZfImpl(Float.POSITIVE_INFINITY, 0.0f));
        assertEquals("float 1^inf re", Float.NaN, h.re(), 0.0f);
        assertEquals("float 1^inf im", Float.NaN, h.im(), 0.0f);
    }

    @Test
    public void testPowOfANanBaseStaysNan() {
        assertZ("(inf,NaN)^2", Double.NaN, Double.NaN, new ZdImpl(Double.POSITIVE_INFINITY, Double.NaN).pow(2.0));
        assertZ("(NaN,0)^2", Double.NaN, Double.NaN, new ZdImpl(Double.NaN, 0.0).pow(2.0));
    }

    @Test
    public void testPowOfOrdinaryBasesIsUnchanged() {
        for (int e = -150; e <= 150; e += 3) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                double re = r * Math.cos(angle(k));
                double im = r * Math.sin(angle(k));
                for (double x : new double[] { 2.0, 0.5, -1.5, 0.0 }) {
                    Zd got = new ZdImpl(re, im).pow(x);
                    Zd want = new ZdImpl(re, im).ln().scale(x).exp();
                    assertEquals("re at 1e" + e, Double.doubleToLongBits(want.re()),
                            Double.doubleToLongBits(got.re()));
                    assertEquals("im at 1e" + e, Double.doubleToLongBits(want.im()),
                            Double.doubleToLongBits(got.im()));
                }
            }
        }
    }

    @Test
    public void testMultiplyingByOneLeavesAnInfiniteValueAlone() {
        double inf = Double.POSITIVE_INFINITY;
        double neg = Double.NEGATIVE_INFINITY;
        assertZ("(-inf,0) scaled by 1", neg, 0.0, new ZdImpl(neg, 0.0).scale(1.0));
        assertZ("(-inf,0) times one", neg, 0.0, new ZdImpl(neg, 0.0).mul(Zd.One()));
        assertZ("(inf,inf) scaled by 1", inf, inf, new ZdImpl(inf, inf).scale(1.0));
        assertZ("(inf,inf) times one", inf, inf, new ZdImpl(inf, inf).mul(Zd.One()));
        Zf f = new ZfImpl(Float.NEGATIVE_INFINITY, 0.0f).scale(1.0f);
        assertEquals("float re", Float.NEGATIVE_INFINITY, f.re(), 0.0f);
        assertEquals("float im", 0.0f, f.im(), 0.0f);
    }

    @Test
    public void testAnInfiniteProductKeepsItsDirection() {
        double inf = Double.POSITIVE_INFINITY;
        double neg = Double.NEGATIVE_INFINITY;
        // the argument of the product is the sum of the arguments
        assertZ("(-1,0)*(inf,inf)", neg, neg, new ZdImpl(-1.0, 0.0).mul(new ZdImpl(inf, inf)));
        assertZ("(1,0)*(-inf,0)", neg, 0.0, new ZdImpl(1.0, 0.0).mul(new ZdImpl(neg, 0.0)));
        assertZ("(1,inf)*(1,-inf)", inf, 0.0, new ZdImpl(1.0, inf).mul(new ZdImpl(1.0, neg)));
        assertZ("(inf,inf)*(inf,inf)", 0.0, inf, new ZdImpl(inf, inf).mul(new ZdImpl(inf, inf)));
        assertZ("(inf,0)*(0,1)", 0.0, inf, new ZdImpl(inf, 0.0).mul(Zd.I()));
        assertZ("(1,1) scaled by -inf", neg, neg, new ZdImpl(1.0, 1.0).scale(neg));
    }

    @Test
    public void testZeroTimesInfinityIsNan() {
        double inf = Double.POSITIVE_INFINITY;
        assertZ("(inf,inf)*(0,0)", Double.NaN, Double.NaN, new ZdImpl(inf, inf).mul(Zd.Zero()));
        assertZ("(0,0)*(inf,inf)", Double.NaN, Double.NaN, new ZdImpl(0.0, 0.0).mul(new ZdImpl(inf, inf)));
        assertZ("(-inf,0) scaled by 0", Double.NaN, Double.NaN,
                new ZdImpl(Double.NEGATIVE_INFINITY, 0.0).scale(0.0));
        Zf f = new ZfImpl(Float.POSITIVE_INFINITY, 0.0f).scale(0.0f);
        assertEquals("float re", Float.NaN, f.re(), 0.0f);
        assertEquals("float im", Float.NaN, f.im(), 0.0f);
    }

    @Test
    public void testMultiplyIsSafeWhenBothOperandsAreTheSameObject() {
        Zd z = new ZdImpl(3.0, 4.0);
        Zd w = new ZdImpl(3.0, 4.0);
        assertZ("aliased", w.copy().mul(w.copy()).re(), w.copy().mul(w.copy()).im(), z.mul(z));
        Zd u = new ZdImpl(1.0, Double.NEGATIVE_INFINITY);
        assertZ("aliased and infinite", Double.NEGATIVE_INFINITY, 0.0, u.mul(u));
    }

    @Test
    public void testAnInfiniteQuotientKeepsItsDirection() {
        double inf = Double.POSITIVE_INFINITY;
        double neg = Double.NEGATIVE_INFINITY;
        // the argument of the quotient is the difference of the arguments
        assertZ("(-inf,inf)/(3,4)", inf, inf, new ZdImpl(neg, inf).div(new ZdImpl(3.0, 4.0)));
        assertZ("(inf,inf)/(3,4)", inf, neg, new ZdImpl(inf, inf).div(new ZdImpl(3.0, 4.0)));
        assertZ("(1,inf)/(1,0)", 0.0, inf, new ZdImpl(1.0, inf).div(Zd.One()));
        assertZ("(-inf,inf)/(1,0)", neg, inf, new ZdImpl(neg, inf).div(Zd.One()));
        Zf f = new ZfImpl(1.0f, Float.POSITIVE_INFINITY).div(Zf.One());
        assertEquals("float re", 0.0f, f.re(), 0.0f);
        assertEquals("float im", Float.POSITIVE_INFINITY, f.im(), 0.0f);
    }

    @Test
    public void testQuotientsWithoutADirection() {
        double inf = Double.POSITIVE_INFINITY;
        Zd w = new ZdImpl(1.0, inf);
        assertZ("inf / inf", Double.NaN, Double.NaN, w.div(w));
        assertZ("inf / NaN", Double.NaN, Double.NaN,
                new ZdImpl(inf, inf).div(new ZdImpl(1.0, Double.NaN)));
        assertZ("finite / inf", 0.0, 0.0, new ZdImpl(3.0, 4.0).div(new ZdImpl(inf, inf)));
        assertZ("zero over zero", Double.NaN, Double.NaN, Zd.Zero().div(Zd.Zero()));
    }

    @Test
    public void testInverseAgreesWithOneOverZ() {
        double inf = Double.POSITIVE_INFINITY;
        double[][] cases = { { 2.0, 3.0 }, { 0.0, 0.0 }, { inf, 1.0 }, { 1.0e-200, 1.0e-200 } };
        for (double[] v : cases) {
            Zd a = new ZdImpl(v[0], v[1]).inv();
            Zd b = Zd.One().div(new ZdImpl(v[0], v[1]));
            assertZ("1/(" + v[0] + "," + v[1] + ")", a.re(), a.im(), b);
        }
    }

    @Test
    public void testDivideIsSafeWhenBothOperandsAreTheSameObject() {
        Zd z = new ZdImpl(3.0, 4.0);
        Zd w = new ZdImpl(3.0, 4.0);
        assertZ("aliased", 1.0, 0.0, z.div(z));
        assertZ("not aliased", 1.0, 0.0, w.div(w.copy()));
    }

    private static void assertSameHash(String what, Zd a, Zd b) {
        assertTrue(what + ": not equal", a.equals(b));
        assertEquals(what + ": equal but hashed differently", a.hashCode(), b.hashCode());
    }

    @Test
    public void testTheTwoZerosHashAlike() {
        assertSameHash("(1,+0) and (1,-0)", new ZdImpl(1.0, 0.0), new ZdImpl(1.0, -0.0));
        assertSameHash("(+0,1) and (-0,1)", new ZdImpl(0.0, 1.0), new ZdImpl(-0.0, 1.0));
        // conj() on a real value is how one walks into this
        assertSameHash("(1,0).conj()", new ZdImpl(1.0, 0.0).conj(), new ZdImpl(1.0, 0.0));
        Zf f = new ZfImpl(1.0f, 0.0f);
        Zf g = new ZfImpl(1.0f, -0.0f);
        assertTrue("float not equal", f.equals(g));
        assertEquals("float equal but hashed differently", f.hashCode(), g.hashCode());
    }

    @Test
    public void testEveryNanHashesAlike() {
        double nan = Double.NaN;
        assertSameHash("(NaN,1) and (NaN,2)", new ZdImpl(nan, 1.0), new ZdImpl(nan, 2.0));
        assertSameHash("(NaN,0) and (0,NaN)", new ZdImpl(nan, 0.0), new ZdImpl(0.0, nan));
        assertSameHash("(NaN,0) and NaN()", new ZdImpl(nan, 0.0), Zd.NaN());
        assertSameHash("(1,NaN) and NaN()", new ZdImpl(1.0, nan), Zd.NaN());
        Zf f = new ZfImpl(Float.NaN, 1.0f);
        Zf g = new ZfImpl(0.0f, Float.NaN);
        assertTrue("float not equal", f.equals(g));
        assertEquals("float equal but hashed differently", f.hashCode(), g.hashCode());
    }

    @Test
    public void testHashCodeIsStable() {
        Zd z = new ZdImpl(3.0, -4.0);
        int first = z.hashCode();
        assertEquals(first, z.hashCode());
        assertEquals(first, z.copy().hashCode());
    }

    @Test
    public void testHashCodeStillSeparatesValues() {
        // the canonicalization must not collapse ordinary values
        java.util.Set<Integer> hashes = new java.util.HashSet<Integer>();
        java.util.Random rnd = new java.util.Random(41L);
        int n = 5000;
        for (int i = 0; i < n; ++i) {
            hashes.add(Integer.valueOf(new ZdImpl(rnd.nextGaussian(), rnd.nextGaussian()).hashCode()));
        }
        assertTrue("only " + hashes.size() + " distinct hashes for " + n + " values", hashes.size() > n - n / 100);
    }

    @Test
    public void testFromPolarWithAnInfiniteRadius() {
        double inf = Double.POSITIVE_INFINITY;
        // radius * sin(0.0) must not turn into NaN
        assertZ("fromPolar(inf, 0)", inf, 0.0, ZdImpl.fromPolar(inf, 0.0));
        assertZ("fromPolar(2, 0)", 2.0, 0.0, ZdImpl.fromPolar(2.0, 0.0));
        assertZ("fromPolar(0, pi)", -0.0, 0.0, ZdImpl.fromPolar(0.0, Math.PI));
        Zf f = ZfImpl.fromPolar(Float.POSITIVE_INFINITY, 0.0f);
        assertEquals("float re", Float.POSITIVE_INFINITY, f.re(), 0.0f);
        assertEquals("float im", 0.0f, f.im(), 0.0f);
    }

    @Test
    public void testFromPolarStillRejectsANegativeRadius() {
        try {
            ZdImpl.fromPolar(-1.0, 0.0);
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // as documented
        }
    }

    @Test
    public void testIsRealIsFalseForNan() {
        double inf = Double.POSITIVE_INFINITY;
        assertTrue("(1,0)", new ZdImpl(1.0, 0.0).isReal());
        assertTrue("(1,-0.0)", new ZdImpl(1.0, -0.0).isReal());
        // the real axis reaches infinity, but NaN is not on it
        assertTrue("(inf,0)", new ZdImpl(inf, 0.0).isReal());
        assertTrue("(NaN,0)", !new ZdImpl(Double.NaN, 0.0).isReal());
        assertTrue("(1,1)", !new ZdImpl(1.0, 1.0).isReal());
        assertTrue("(1,NaN)", !new ZdImpl(1.0, Double.NaN).isReal());
        assertTrue("float (NaN,0)", !new ZfImpl(Float.NaN, 0.0f).isReal());
        assertTrue("float (1,0)", new ZfImpl(1.0f, 0.0f).isReal());
    }

    @Test
    public void testAbsOnTheBranchThatWasDead() {
        // |im| >= |re| and im != 0 is the branch the dead test guarded
        assertEquals("(3,4)", 5.0, new ZdImpl(3.0, 4.0).abs(), 1.0e-15);
        assertEquals("static (3,4)", 5.0, ZdImpl.abs(3.0, 4.0), 1.0e-15);
        assertEquals("(0,-1)", 1.0, new ZdImpl(0.0, -1.0).abs(), 0.0);
        assertEquals("(1e-320,1e-320)", ZdImpl.abs(1.0e-320, 1.0e-320), new ZdImpl(1.0e-320, 1.0e-320).abs(), 0.0);
        assertTrue("(NaN,1)", Double.isNaN(ZdImpl.abs(Double.NaN, 1.0)));
        assertTrue("(1,NaN)", Double.isNaN(ZdImpl.abs(1.0, Double.NaN)));
        assertEquals("float (3,4)", 5.0f, ZfImpl.abs(3.0f, 4.0f), 1.0e-6f);
        assertEquals("(0,0)", 0.0, ZdImpl.abs(0.0, 0.0), 0.0);
    }

    @Test
    public void testInfinityTimesNanIsNan() {
        double inf = Double.POSITIVE_INFINITY;
        double nan = Double.NaN;
        // zeroing a NaN component must not turn the product into a zero
        assertZ("(inf,inf)*(NaN,NaN)", nan, nan, new ZdImpl(inf, inf).mul(new ZdImpl(nan, nan)));
        assertZ("(inf,inf)*(NaN,0)", nan, nan, new ZdImpl(inf, inf).mul(new ZdImpl(nan, 0.0)));
        assertZ("(inf,0)*(0,NaN)", nan, nan, new ZdImpl(inf, 0.0).mul(new ZdImpl(0.0, nan)));
        assertZ("(inf,inf) scaled by NaN", nan, nan, new ZdImpl(inf, inf).scale(nan));
        // but a single NaN component leaves the direction intact
        assertZ("(-inf,-inf)*(1,NaN)", Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY,
                new ZdImpl(-inf, -inf).mul(new ZdImpl(1.0, nan)));
        Zf f = new ZfImpl(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY).scale(Float.NaN);
        assertEquals("float re", Float.NaN, f.re(), 0.0f);
        assertEquals("float im", Float.NaN, f.im(), 0.0f);
    }

    @Test
    public void testSqrtSquaredIsTheOriginal() {
        for (int e = MIN_EXP_D; e <= MAX_EXP_D; ++e) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                Zd z = new ZdImpl(r * Math.cos(angle(k)), r * Math.sin(angle(k)));
                Zd s = z.copy().sqrt();
                assertTrue("1e" + e, relative(s.copy().mul(s.copy()), z) <= 1.0e-15);
            }
        }
    }

    @Test
    public void testSqrtIsThePrincipalValue() {
        for (int e = MIN_EXP_D; e <= MAX_EXP_D; ++e) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                double b = r * Math.sin(angle(k));
                Zd s = new ZdImpl(r * Math.cos(angle(k)), b).sqrt();
                assertTrue("real part at 1e" + e, s.re() >= 0.0);
                assertTrue("sign at 1e" + e, Math.signum(s.im()) == Math.signum(b));
            }
        }
    }

    @Test
    public void testSqrtFarOutOfRange() {
        // the squared modulus over- and underflows here, abs() does not
        for (double r : new double[] { 1.0e+300, 1.0e+200, 1.0e-200, 1.0e-300 }) {
            Zd z = new ZdImpl(r * Math.cos(0.7), r * Math.sin(0.7));
            Zd s = z.copy().sqrt();
            assertTrue("finite at " + r, !Double.isInfinite(s.re()) && !Double.isNaN(s.re()));
            assertTrue("value at " + r, relative(s.copy().mul(s.copy()), z) <= 1.0e-15);
        }
    }

    @Test
    public void testSqrtEdgeCases() {
        double inf = Double.POSITIVE_INFINITY;
        double nan = Double.NaN;
        assertZ("(4,0)", 2.0, 0.0, new ZdImpl(4.0, 0.0).sqrt());
        assertZ("(-4,0)", 0.0, 2.0, new ZdImpl(-4.0, 0.0).sqrt());
        // copySign carries the branch cut
        assertZ("(-4,-0.0)", 0.0, -2.0, new ZdImpl(-4.0, -0.0).sqrt());
        assertZ("(0,0)", 0.0, 0.0, new ZdImpl(0.0, 0.0).sqrt());
        assertZ("(-0.0,0.0)", 0.0, 0.0, new ZdImpl(-0.0, 0.0).sqrt());
        assertZ("(inf,1)", inf, 0.0, new ZdImpl(inf, 1.0).sqrt());
        assertZ("(-inf,1)", 0.0, inf, new ZdImpl(-inf, 1.0).sqrt());
        assertZ("(inf,inf)", inf, inf, new ZdImpl(inf, inf).sqrt());
        assertZ("(1,inf)", inf, inf, new ZdImpl(1.0, inf).sqrt());
        assertZ("(1,-inf)", inf, -inf, new ZdImpl(1.0, -inf).sqrt());
        assertZ("(NaN,1)", nan, nan, new ZdImpl(nan, 1.0).sqrt());
        assertZ("(1,NaN)", nan, nan, new ZdImpl(1.0, nan).sqrt());
    }

    @Test
    public void testSqrtInSinglePrecision() {
        Zf a = new ZfImpl(-4.0f, 0.0f).sqrt();
        assertEquals("(-4,0) re", 0.0f, a.re(), 0.0f);
        assertEquals("(-4,0) im", 2.0f, a.im(), 1.0e-6f);
        Zf b = new ZfImpl(1.0f, Float.POSITIVE_INFINITY).sqrt();
        assertEquals("(1,inf) re", Float.POSITIVE_INFINITY, b.re(), 0.0f);
        assertEquals("(1,inf) im", Float.POSITIVE_INFINITY, b.im(), 0.0f);
        Zf c = new ZfImpl(0.0f, 0.0f).sqrt();
        assertEquals("(0,0) re", 0.0f, c.re(), 0.0f);
        assertEquals("(0,0) im", 0.0f, c.im(), 0.0f);
        // the squared modulus would overflow a float here
        Zf d = new ZfImpl(1.0e30f, 1.0e30f).sqrt();
        assertTrue("(1e30,1e30) finite", !Float.isInfinite(d.re()) && !Float.isNaN(d.re()));
    }

    @Test
    public void testInvKeepsTheInfinityConvention() {
        Zd zero = new ZdImpl(0.0, 0.0).inv();
        assertEquals(Double.POSITIVE_INFINITY, zero.re(), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, zero.im(), 0.0);
        Zd inf = new ZdImpl(Double.POSITIVE_INFINITY, 0.0).inv();
        assertEquals(0.0, inf.re(), 0.0);
        assertEquals(0.0, inf.im(), 0.0);
        Zf zeroF = new ZfImpl(0.0f, 0.0f).inv();
        assertEquals(Float.POSITIVE_INFINITY, zeroF.re(), 0.0f);
        assertEquals(Float.POSITIVE_INFINITY, zeroF.im(), 0.0f);
        Zf infF = new ZfImpl(Float.POSITIVE_INFINITY, 0.0f).inv();
        assertEquals(0.0f, infF.re(), 0.0f);
        assertEquals(0.0f, infF.im(), 0.0f);
    }
}
