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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.fail;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

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


    private static void assertZf(String what, float wantRe, float wantIm, Zf got) {
        assertEquals(what + " re", wantRe, got.re(), 0.0f);
        assertEquals(what + " im", wantIm, got.im(), 0.0f);
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
    public void testTheTwoZerosAreToldApart() {
        // the branch cuts read the sign of a zero, so equals does too
        assertDifferent("(1,+0) and (1,-0)", new ZdImpl(1.0, 0.0), new ZdImpl(1.0, -0.0));
        assertDifferent("(+0,1) and (-0,1)", new ZdImpl(0.0, 1.0), new ZdImpl(-0.0, 1.0));
        // conj() on a real value is how one walks into this
        assertDifferent("(1,0).conj()", new ZdImpl(1.0, 0.0).conj(), new ZdImpl(1.0, 0.0));
        // the four zeros are four values with four hashes
        Zd[] zeros = { new ZdImpl(0.0, 0.0), new ZdImpl(-0.0, -0.0), new ZdImpl(0.0, -0.0),
                new ZdImpl(-0.0, 0.0) };
        for (int i = 0; i < zeros.length; ++i) {
            for (int j = i + 1; j < zeros.length; ++j) {
                assertDifferent("zero " + i + " and " + j, zeros[i], zeros[j]);
            }
        }
        Zf f = new ZfImpl(1.0f, 0.0f);
        Zf g = new ZfImpl(1.0f, -0.0f);
        assertFalse("float equal", f.equals(g));
        assertTrue("float hashed alike", f.hashCode() != g.hashCode());
        Zf[] fz = { new ZfImpl(0.0f, 0.0f), new ZfImpl(-0.0f, -0.0f), new ZfImpl(0.0f, -0.0f),
                new ZfImpl(-0.0f, 0.0f) };
        for (int i = 0; i < fz.length; ++i) {
            for (int j = i + 1; j < fz.length; ++j) {
                assertFalse("float zero " + i + " and " + j, fz[i].equals(fz[j]));
                assertTrue("float zero " + i + " and " + j + " hashed alike",
                        fz[i].hashCode() != fz[j].hashCode());
            }
        }
    }

    private static void assertDifferent(String what, Zd a, Zd b) {
        assertFalse(what + ": equal", a.equals(b));
        assertTrue(what + ": hashed alike", a.hashCode() != b.hashCode());
    }

    @Test
    public void testHashCodeSpreadsARegularGrid() {
        // the mixing this replaced folded a regular grid onto 3.9 percent of
        // its values; regular grids are what numerical code produces
        java.util.HashSet<Integer> d = new java.util.HashSet<Integer>();
        java.util.HashSet<Integer> f = new java.util.HashSet<Integer>();
        int n = 0;
        for (int i = -160; i <= 160; ++i) {
            for (int j = -160; j <= 160; ++j) {
                ++n;
                d.add(new ZdImpl(i * 0.25, j * 0.25).hashCode());
                f.add(new ZfImpl(i * 0.25f, j * 0.25f).hashCode());
            }
        }
        assertTrue("double: " + d.size() + " of " + n, d.size() > 0.99 * n);
        assertTrue("float: " + f.size() + " of " + n, f.size() > 0.99 * n);
    }

    @Test
    public void testTheEqualsContract() {
        double inf = Double.POSITIVE_INFINITY;
        double nan = Double.NaN;
        double[] vals = { 0.0, -0.0, 1.0, -1.0, 2.5, inf, -inf, nan, 4.9e-324, 1.0e300 };
        List<Zd> zs = new ArrayList<Zd>();
        for (int i = 0; i < vals.length; ++i) {
            for (int j = 0; j < vals.length; ++j) {
                zs.add(new ZdImpl(vals[i], vals[j]));
            }
        }
        for (Zd p : zs) {
            assertTrue("reflexive: " + p, p.equals(p));
            assertFalse("null: " + p, p.equals(null));
            assertFalse("foreign class: " + p, p.equals("z"));
            for (Zd q : zs) {
                assertEquals("symmetric: " + p + " / " + q, p.equals(q), q.equals(p));
                if (p.equals(q)) {
                    assertEquals("hashCode: " + p + " / " + q, p.hashCode(), q.hashCode());
                }
            }
        }
        for (Zd p : zs) {
            for (Zd q : zs) {
                if (!p.equals(q)) {
                    continue;
                }
                for (Zd r : zs) {
                    if (q.equals(r)) {
                        assertTrue("transitive: " + p + " / " + q + " / " + r, p.equals(r));
                    }
                }
            }
        }
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

    /** the sqrt tests reach further out than the shared ensemble */
    private static final int SQRT_MIN_EXP_D = -323;
    private static final int SQRT_MAX_EXP_D = 308;
    private static final int SQRT_MIN_EXP_F = -45;
    private static final int SQRT_MAX_EXP_F = 38;

    @Test
    public void testSqrtOnTheRealAxisIsMathSqrt() {
        // on the axis the modulus is |x|, so t is Math.sqrt(x) and must agree
        // to the last bit; Math.sqrt has nothing in common with the code
        for (int e = SQRT_MIN_EXP_D; e <= SQRT_MAX_EXP_D; ++e) {
            for (double m : new double[] { 1.0, 2.5, 7.3 }) {
                double x = m * Math.pow(10.0, e);
                if (Double.isInfinite(x) || x == 0.0) {
                    continue;
                }
                assertZ("sqrt(" + x + ")", Math.sqrt(x), 0.0, new ZdImpl(x, 0.0).sqrt());
                assertZ("sqrt(-" + x + ")", 0.0, Math.sqrt(x), new ZdImpl(-x, 0.0).sqrt());
            }
        }
        for (int e = SQRT_MIN_EXP_F; e <= SQRT_MAX_EXP_F; ++e) {
            for (double m : new double[] { 1.0, 2.5, 7.3 }) {
                float x = (float) (m * Math.pow(10.0, e));
                if (Float.isInfinite(x) || x == 0.0f) {
                    continue;
                }
                assertEquals("float sqrt(" + x + ")", (float) Math.sqrt(x),
                        new ZfImpl(x, 0.0f).sqrt().re(), 0.0f);
            }
        }
    }

    @Test
    public void testSqrtStaysFiniteAtBothEnds() {
        // the sum |re| + |z| overflows above a modulus of 9e307 and turns
        // subnormal below 2e-308; both ends must still come out of the range
        for (int e = SQRT_MIN_EXP_D; e <= SQRT_MAX_EXP_D; ++e) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                double x = r * Math.cos(angle(k));
                double y = r * Math.sin(angle(k));
                if (Double.isInfinite(x) || Double.isInfinite(y)) {
                    continue;
                }
                Zd s = new ZdImpl(x, y).sqrt();
                assertTrue("finite at 1e" + e, !Double.isInfinite(s.re()) && !Double.isNaN(s.re())
                        && !Double.isInfinite(s.im()) && !Double.isNaN(s.im()));
                assertTrue("not zero at 1e" + e, s.re() != 0.0 || s.im() != 0.0);
            }
        }
        for (int e = SQRT_MIN_EXP_F; e <= SQRT_MAX_EXP_F; ++e) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                float x = (float) (r * Math.cos(angle(k)));
                float y = (float) (r * Math.sin(angle(k)));
                if (Float.isInfinite(x) || Float.isInfinite(y)) {
                    continue;
                }
                Zf s = new ZfImpl(x, y).sqrt();
                assertTrue("float finite at 1e" + e, !Float.isInfinite(s.re()) && !Float.isNaN(s.re())
                        && !Float.isInfinite(s.im()) && !Float.isNaN(s.im()));
                assertTrue("float not zero at 1e" + e, s.re() != 0.0f || s.im() != 0.0f);
            }
        }
        assertEquals("sqrt(9e307)", 9.486832980505138E153, new ZdImpl(9.0e307, 0.0).sqrt().re(), 0.0);
        assertEquals("float sqrt(3.4e38)", (float) Math.sqrt(3.4e38f),
                new ZfImpl(3.4e38f, 0.0f).sqrt().re(), 0.0f);
    }

    @Test
    public void testSqrtWithVeryUnequalComponents() {
        // where |b| is far below a > 0 the root is (sqrt(a), b/(2*sqrt(a))),
        // and Math.sqrt says so to the last bit. Scaling the argument by more
        // than a factor of four would push b/(2t) into the subnormal range and
        // lose it here, which the ensemble above never reaches.
        for (int ea = -300; ea <= SQRT_MAX_EXP_D; ++ea) {
            double a = 1.7 * Math.pow(10.0, ea);
            if (Double.isInfinite(a)) {
                continue;
            }
            for (int d = 20; d <= 620; d += 20) {
                double b = 3.1 * Math.pow(10.0, ea - d);
                double wantIm = b / (2.0 * Math.sqrt(a));
                if (b == 0.0 || wantIm == 0.0) {
                    continue;
                }
                Zd s = new ZdImpl(a, b).sqrt();
                assertEquals("sqrt re at (" + a + "," + b + ")", Math.sqrt(a), s.re(), 0.0);
                // one ulp of a subnormal is all the factor of four may cost here
                assertTrue("sqrt im at (" + a + "," + b + ") is " + s.im(),
                        Math.abs(s.im() - wantIm) <= 1.0e-13 * Math.abs(wantIm));
            }
        }
        for (int ea = -38; ea <= SQRT_MAX_EXP_F; ++ea) {
            float a = (float) (1.7 * Math.pow(10.0, ea));
            if (Float.isInfinite(a) || a == 0.0f) {
                continue;
            }
            for (int d = 4; d <= 40; d += 2) {
                float b = (float) (3.1 * Math.pow(10.0, ea - d));
                double wantIm = b / (2.0 * Math.sqrt(a));
                if (b == 0.0f || Math.abs(wantIm) < Float.MIN_NORMAL) {
                    continue;
                }
                Zf s = new ZfImpl(a, b).sqrt();
                assertTrue("float sqrt re at (" + a + "," + b + ") is " + s.re(),
                        Math.abs(s.re() - Math.sqrt(a)) <= 1.0e-7 * Math.sqrt(a));
                assertTrue("float sqrt im at (" + a + "," + b + ") is " + s.im(),
                        Math.abs(s.im() - wantIm) <= 1.0e-7 * Math.abs(wantIm));
            }
        }
    }

    @Test
    public void testSqrtInTheSubnormalBand() {
        // the expected values come from 120 digit arithmetic, because squaring
        // the root back underflows down here
        subnormalRoot(1.0e-320, 1.0e-320, 1.0986779977260263990E-160, 4.5508732733903661602E-161);
        subnormalRoot(1.0e-315, -2.0e-316, 3.1778954514264884631E-158, -3.1467366489798094928E-159);
        subnormalRoot(-1.0e-310, 3.0e-311, 1.4837562281427940884E-156, 1.0109477362581718829E-155);
        subnormalRootF(1.4e-45f, 1.4e-45f, 4.1128054643427787981E-23, 1.7035798027329537504E-23);
        subnormalRootF(1.0e-42f, -3.0e-43f, 1.0111942368118900672E-21, -1.4827906471805593517E-22);
        subnormalRootF(-1.0e-40f, 2.0e-41f, 9.9505519252915438348E-22, 1.0049357981847737337E-20);
        // three points pinned to the bit, where taking the quotient of the
        // modulus in float instead of double moves the answer away from the
        // 120 digit value by a fraction of an ulp
        assertZf("sqrt(-6.1964989e9, -1.12513096e10)", 57655.39f, -97573.78f,
                new ZfImpl(-6.1964989E9f, -1.12513096E10f).sqrt());
        assertZf("sqrt(2.10404086e10, 3.30625167e10)", 173536.83f, 95260.805f,
                new ZfImpl(2.10404086E10f, 3.30625167E10f).sqrt());
        assertZf("sqrt(-43.956238, 68.04957)", 4.3043847f, 7.90468f,
                new ZfImpl(-43.956238f, 68.04957f).sqrt());
    }

    private static void subnormalRoot(double x, double y, double wantRe, double wantIm) {
        Zd got = new ZdImpl(x, y).sqrt();
        assertTrue("sqrt re at (" + x + "," + y + "): want " + wantRe + ", got " + got.re(),
                Math.abs(got.re() - wantRe) <= 1.0e-15 * Math.abs(wantRe));
        assertTrue("sqrt im at (" + x + "," + y + "): want " + wantIm + ", got " + got.im(),
                Math.abs(got.im() - wantIm) <= 1.0e-15 * Math.abs(wantIm));
    }

    private static void subnormalRootF(float x, float y, double wantRe, double wantIm) {
        Zf got = new ZfImpl(x, y).sqrt();
        assertTrue("float sqrt re at (" + x + "," + y + "): want " + wantRe + ", got " + got.re(),
                Math.abs(got.re() - wantRe) <= 1.0e-6 * Math.abs(wantRe));
        assertTrue("float sqrt im at (" + x + "," + y + "): want " + wantIm + ", got " + got.im(),
                Math.abs(got.im() - wantIm) <= 1.0e-6 * Math.abs(wantIm));
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
