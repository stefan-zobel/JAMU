/*
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

import org.junit.Assert;
import org.junit.Test;

public final class ZdImplBasicsTest {

    private static final double inf = Double.POSITIVE_INFINITY;
    private static final double neginf = Double.NEGATIVE_INFINITY;
    private static final double nan = Double.NaN;
    private static final double pi = Math.PI;
    private static Zd oneInf() {return new ZdImpl(1, inf);}
    private static Zd oneNegInf() {return new ZdImpl(1, neginf);}
    private static Zd infOne() {return new ZdImpl(inf, 1);}
    private static Zd infZero() {return new ZdImpl(inf, 0);}
    private static Zd infNegInf() {return new ZdImpl(inf, neginf);}
    private static Zd infInf() {return new ZdImpl(inf, inf);}
    private static Zd negInfInf() {return new ZdImpl(neginf, inf);}
    private static Zd negInfZero() {return new ZdImpl(neginf, 0);}
    private static Zd negInfOne() {return new ZdImpl(neginf, 1);}
    private static Zd negInfNegInf() {return new ZdImpl(neginf, neginf);}
    private static Zd oneNaN() {return new ZdImpl(1, nan);}
    private static Zd zeroInf() {return new ZdImpl(0, inf);}
    private static Zd zeroNaN() {return new ZdImpl(0, nan);}
    private static Zd nanZero() {return new ZdImpl(nan, 0);}

    @Test
    public void testConstructor() {
        Zd z = new ZdImpl(3.0, 4.0);
        Assert.assertEquals(3.0, z.re(), 1.0e-5);
        Assert.assertEquals(4.0, z.im(), 1.0e-5);
    }

    @Test
    public void testConstructorNaN() {
        Zd z = new ZdImpl(3.0, Double.NaN);
        Assert.assertTrue(z.isNan());

        z = new ZdImpl(nan, 4.0);
        Assert.assertTrue(z.isNan());

        z = new ZdImpl(3.0, 4.0);
        Assert.assertFalse(z.isNan());
    }

    @Test
    public void testAbs() {
        Zd z = new ZdImpl(3.0, 4.0);
        Assert.assertEquals(5.0, z.abs(), 1.0e-5);
    }

    @Test
    public void testAbsNaN() {
        Assert.assertTrue(Double.isNaN(Zd.NaN().abs()));
        Zd z = new ZdImpl(nan, nan);
        double abs = z.abs();
        Assert.assertTrue(Double.isNaN(abs));
    }

    @Test
    public void testAbsNaNInf() {
        Assert.assertTrue(Double.isInfinite(Zd.Inf().abs()));
        Zd z = new ZdImpl(inf, nan);
        double abs = z.abs();
        Assert.assertTrue(Double.isInfinite(abs));
    }

    @Test
    public void testAbsInfinite() {
        Zd z = new ZdImpl(inf, 0);
        Assert.assertEquals(inf, z.abs(), 0);
        z = new ZdImpl(0, neginf);
        Assert.assertEquals(inf, z.abs(), 0);
        z = new ZdImpl(inf, neginf);
        Assert.assertEquals(inf, z.abs(), 0);
    }

    @Test
    public void testAdd() {
        Zd x = new ZdImpl(3.0, 4.0);
        Zd y = new ZdImpl(5.0, 6.0);
        Zd z = x.add(y);
        Assert.assertEquals(8.0, z.re(), 1.0e-5);
        Assert.assertEquals(10.0, z.im(), 1.0e-5);
    }

    @Test
    public void testAddNaN() {
        Zd x = new ZdImpl(3.0, 4.0);
        Zd z = x.add(Zd.NaN());
        Assert.assertEquals(Zd.NaN(), z);
        z = new ZdImpl(1, nan);
        Zd w = x.add(z);
        Assert.assertEquals(Zd.NaN(), w);
    }

    @Test
    public void testAddInf() {
        Zd x = new ZdImpl(1, 1);
        Zd z = new ZdImpl(inf, 0);
        Zd w = x.add(z);
        Assert.assertEquals(w.im(), 1, 0);
        Assert.assertEquals(inf, w.re(), 0);

        x = new ZdImpl(neginf, 0);
        Assert.assertTrue(Double.isNaN(x.add(z).re()));
    }

    @Test
    public void testConj() {
        Zd x = new ZdImpl(3.0, 4.0);
        Zd z = x.conj();
        Assert.assertEquals(3.0, z.re(), 1.0e-5);
        Assert.assertEquals(-4.0, z.im(), 1.0e-5);
    }

    @Test
    public void testConjugateNaN() {
        Zd z = Zd.NaN().conj();
        Assert.assertTrue(z.isNan());
    }

    @Test
    public void testConjugateInfinite() {
        Zd z = new ZdImpl(0, inf);
        Assert.assertEquals(neginf, z.conj().im(), 0);
        z = new ZdImpl(0, neginf);
        Assert.assertEquals(inf, z.conj().im(), 0);
    }

    @Test
    public void testDiv() {
        Zd x = new ZdImpl(3.0, 4.0);
        Zd y = new ZdImpl(5.0, 6.0);
        Zd z = x.div(y);
        Assert.assertEquals(39.0 / 61.0, z.re(), 1.0e-5);
        Assert.assertEquals(2.0 / 61.0, z.im(), 1.0e-5);
    }

    @Test
    public void testDivideReal() {
        Zd x = new ZdImpl(2d, 3d);
        Zd y = new ZdImpl(2d, 0d);
        Assert.assertEquals(new ZdImpl(1d, 1.5), x.div(y));
    }

    @Test
    public void testDivideImaginary() {
        Zd x = new ZdImpl(2d, 3d);
        Zd y = new ZdImpl(0d, 2d);
        Assert.assertEquals(new ZdImpl(1.5d, -1d), x.div(y));
    }

    @Test
    public void testDivideInf() {
        Zd w = new ZdImpl(neginf, inf);
        TestUtils.assertEquals(Zd.Zero(), new ZdImpl(3, 4).div(w), 10e-12);

        Zd x = new ZdImpl(3, 4);
        Zd z = w.div(x);
        Assert.assertTrue(Double.isNaN(z.re()));
        Assert.assertEquals(inf, z.im(), 0);

        w = new ZdImpl(inf, inf);
        z = w.div(x);
        Assert.assertTrue(Double.isNaN(z.im()));
        Assert.assertEquals(inf, z.re(), 0);

        w = new ZdImpl(1, inf);
        z = w.div(w);
        Assert.assertTrue(Double.isNaN(z.re()));
        Assert.assertTrue(Double.isNaN(z.im()));
    }

    @Test
    public void testDivideZero() {
        Zd x = new ZdImpl(3.0, 4.0);
        Zd z = x.div(Zd.Zero());
        Assert.assertEquals(z, Zd.NaN());
    }

    @Test
    public void testDivideZeroZero() {
        Zd x = new ZdImpl(0.0, 0.0);
        Zd z = x.div(Zd.Zero());
        Assert.assertEquals(z, Zd.NaN());
    }

    @Test
    public void testDivideNaN() {
        Zd x = new ZdImpl(3.0, 4.0);
        Zd z = x.div(Zd.NaN());
        Assert.assertTrue(z.isNan());
    }

    @Test
    public void testDivideNaNInf() {
       Zd z = oneInf().div(Zd.One());
       Assert.assertTrue(Double.isNaN(z.re()));
       Assert.assertEquals(inf, z.im(), 0);

       z = negInfNegInf().div(oneNaN());
       Assert.assertTrue(Double.isNaN(z.re()));
       Assert.assertTrue(Double.isNaN(z.im()));

       z = negInfInf().div(Zd.One());
       Assert.assertTrue(Double.isNaN(z.re()));
       Assert.assertTrue(Double.isNaN(z.im()));
    }

    @Test
    public void testInv() {
        Zd z = new ZdImpl(5.0, 6.0);
        Zd act = z.inv();
        double expRe = 5.0 / 61.0;
        double expIm = -6.0 / 61.0;
        Assert.assertEquals(expRe, act.re(), Math.ulp(expRe));
        Assert.assertEquals(expIm, act.im(), Math.ulp(expIm));
    }

    @Test
    public void testInvReal() {
        Zd z = new ZdImpl(-2.0, 0.0);
        Assert.assertEquals(new ZdImpl(-0.5, 0.0), z.inv());
    }

    @Test
    public void testInvImaginary() {
        Zd z = new ZdImpl(0.0, -2.0);
        Assert.assertEquals(new ZdImpl(0.0, 0.5), z.inv());
    }

    @Test
    public void testInvInf() {
        Zd z = new ZdImpl(neginf, inf);
        Assert.assertTrue(z.inv().equals(Zd.Zero()));

        z = new ZdImpl(1, inf).inv();
        Assert.assertEquals(z, Zd.Zero());
    }

    @Test
    public void testInvZero() {
        Assert.assertEquals(Zd.Zero().inv(), Zd.Inf());
    }

    @Test
    public void testInvNaN() {
        Assert.assertTrue(Zd.NaN().inv().isNan());
    }

    @Test
    public void testMul() {
        Zd x = new ZdImpl(3.0, 4.0);
        Zd y = new ZdImpl(5.0, 6.0);
        Zd z = x.mul(y);
        Assert.assertEquals(-9.0, z.re(), 1.0e-5);
        Assert.assertEquals(38.0, z.im(), 1.0e-5);
    }

    @Test
    public void testMul2() {
        Zd x = new ZdImpl(3.0, 4.0);
        Zd y = x;
        Zd z = x.mul(y);
        Assert.assertEquals(-7.0, z.re(), 1.0e-5);
        Assert.assertEquals(24.0, z.im(), 1.0e-5);
    }

    @Test
    public void testMultiplyNaN() {
        Zd x = new ZdImpl(3.0, 4.0);
        Zd z = x.mul(Zd.NaN());
        Assert.assertEquals(Zd.NaN(), z);
        z = Zd.NaN().scale(5);
        Assert.assertEquals(Zd.NaN(), z);
    }

    @Test
    public void testMultiplyInfInf() {
        Assert.assertTrue(infInf().mul(infInf()).isInfinite());
    }

    @Test
    public void testMultiplyNaNInf() {
        Zd z = new ZdImpl(1,1);
        Zd w = z.mul(infOne());
        Assert.assertEquals(w.re(), inf, 0);
        Assert.assertEquals(w.im(), inf, 0);

        // multiplications with infinity 
        Assert.assertTrue(new ZdImpl( 1,0).mul(infInf()).equals(Zd.Inf()));
        Assert.assertTrue(new ZdImpl(-1,0).mul(infInf()).equals(Zd.Inf()));
        Assert.assertTrue(new ZdImpl( 1,0).mul(negInfZero()).equals(Zd.Inf()));

        w = oneInf().mul(oneNegInf());
        Assert.assertEquals(w.re(), inf, 0);
        Assert.assertEquals(w.im(), inf, 0);

        w = negInfNegInf().mul(oneNaN());
        // TODO: better use isNaN()?
//        Assert.assertTrue(Double.isNaN(w.re()));
//        Assert.assertTrue(Double.isNaN(w.im()));
        Assert.assertTrue(Double.isInfinite(w.re()));
        Assert.assertTrue(Double.isInfinite(w.im()));

        z = new ZdImpl(1, neginf);
        TestUtils.assertSame(Zd.Inf(), z.mul(z));
    }

    @Test
    public void testScale() {
        Zd x = new ZdImpl(3.0, 4.0);
        double yDouble = 2.0;
        Zd yZdImpl = new ZdImpl(yDouble);
        Assert.assertEquals(x.mul(yZdImpl), x.scale(yDouble));
        int zInt = -5;
        Zd zZdImpl = new ZdImpl(zInt);
        Assert.assertEquals(x.mul(zZdImpl), x.scale(zInt));
    }

    @Test
    public void testScaleNaN() {
        Zd x = new ZdImpl(3.0, 4.0);
        double yDouble = Double.NaN;
        Zd yZdImpl = new ZdImpl(yDouble);
        Assert.assertEquals(x.mul(yZdImpl), x.scale(yDouble));
    }

    @Test
    public void testScaleInf() {
        Zd x = new ZdImpl(1, 1);
        double yDouble = Double.POSITIVE_INFINITY;
        Zd yZdImpl = new ZdImpl(yDouble);
        Assert.assertEquals(x.mul(yZdImpl), x.scale(yDouble));

        yDouble = Double.NEGATIVE_INFINITY;
        yZdImpl = new ZdImpl(yDouble);
        Assert.assertEquals(x.mul(yZdImpl), x.scale(yDouble));
    }

    @Test
    public void testNeg() {
        Zd x = new ZdImpl(3.0, 4.0);
        Zd z = x.neg();
        Assert.assertEquals(-3.0, z.re(), 1.0e-5);
        Assert.assertEquals(-4.0, z.im(), 1.0e-5);
    }

    @Test
    public void testNegateNaN() {
        Zd z = Zd.NaN().neg();
        Assert.assertTrue(z.isNan());
    }

    @Test
    public void testSub() {
        Zd x = new ZdImpl(3.0, 4.0);
        Zd y = new ZdImpl(5.0, 6.0);
        Zd z = x.sub(y);
        Assert.assertEquals(-2.0, z.re(), 1.0e-5);
        Assert.assertEquals(-2.0, z.im(), 1.0e-5);
    }

    @Test
    public void testSubtractNaN() {
        Zd x = new ZdImpl(3.0, 4.0);
        Zd z = x.sub(Zd.NaN());
        Assert.assertEquals(Zd.NaN(), z);
        z = new ZdImpl(1, nan);
        Zd w = x.sub(z);
        Assert.assertEquals(Zd.NaN(), w);
    }

    @Test
    public void testSubtractInf() {
        Zd x = new ZdImpl(1, 1);
        Zd z = new ZdImpl(neginf, 0);
        Zd w = x.sub(z);
        Assert.assertEquals(w.im(), 1, 0);
        Assert.assertEquals(inf, w.re(), 0);

        x = new ZdImpl(neginf, 0);
        Assert.assertTrue(Double.isNaN(x.sub(z).re()));
    }

    @Test
    public void testEqualsNull() {
        Zd x = new ZdImpl(3.0, 4.0);
        Assert.assertFalse(x.equals(null));
    }

    @Test
    public void testEqualsClass() {
        Zd x = new ZdImpl(3.0, 4.0);
        Assert.assertFalse(x.equals(this));
    }

    @Test
    public void testEqualsSame() {
        Zd x = new ZdImpl(3.0, 4.0);
        Assert.assertTrue(x.equals(x));
    }

    @Test
    public void testEqualsTrue() {
        Zd x = new ZdImpl(3.0, 4.0);
        Zd y = new ZdImpl(3.0, 4.0);
        Assert.assertTrue(x.equals(y));
    }

    @Test
    public void testEqualsRealDifference() {
        Zd x = new ZdImpl(0.0, 0.0);
        Zd y = new ZdImpl(0.0 + Double.MIN_VALUE, 0.0);
        Assert.assertFalse(x.equals(y));
    }

    @Test
    public void testEqualsImaginaryDifference() {
        Zd x = new ZdImpl(0.0, 0.0);
        Zd y = new ZdImpl(0.0, 0.0 + Double.MIN_VALUE);
        Assert.assertFalse(x.equals(y));
    }

    @Test
    public void testEqualsNaN() {
        Zd realNaN = new ZdImpl(Double.NaN, 0.0);
        Zd imaginaryNaN = new ZdImpl(0.0, Double.NaN);
        Zd complexNaN = Zd.NaN();
        Assert.assertTrue(realNaN.equals(imaginaryNaN));
        Assert.assertTrue(imaginaryNaN.equals(complexNaN));
        Assert.assertTrue(realNaN.equals(complexNaN));
    }

    @Test
    public void testExp() {
        Zd z = new ZdImpl(3, 4);
        Zd expected = new ZdImpl(-13.12878, -15.20078);
        TestUtils.assertEquals(expected, z.exp(), 1.0e-5);
        TestUtils.assertEquals(Zd.One(),
                Zd.Zero().exp(), 10e-12);
        Zd iPi = Zd.I().mul(new ZdImpl(pi, 0));
        TestUtils.assertEquals(Zd.One().neg(),
                iPi.exp(), 10e-12);
    }

    @Test
    public void testExpNaN() {
        Assert.assertTrue(Zd.NaN().exp().isNan());
    }

    @Test
    public void testExpInf() {
        TestUtils.assertSame(Zd.NaN(), oneInf().exp());
        TestUtils.assertSame(Zd.NaN(), oneNegInf().exp());
        TestUtils.assertSame(infInf(), infOne().exp());
        TestUtils.assertSame(Zd.Zero(), negInfOne().exp());
        TestUtils.assertSame(Zd.NaN(), infInf().exp());
        TestUtils.assertSame(Zd.NaN(), infNegInf().exp());
        TestUtils.assertSame(Zd.NaN(), negInfInf().exp());
        TestUtils.assertSame(Zd.NaN(), negInfNegInf().exp());
    }

    @Test
    public void testLn() {
        Zd z = new ZdImpl(3, 4);
        Zd expected = new ZdImpl(1.60944, 0.927295);
        TestUtils.assertEquals(expected, z.ln(), 1.0e-5);
    }

    @Test
    public void testLogNaN() {
        Assert.assertTrue(Zd.NaN().ln().isNan());
    }

    @Test
    public void testLogInf() {
        TestUtils.assertEquals(new ZdImpl(inf, pi / 2),
                oneInf().ln(), 10e-12);
        TestUtils.assertEquals(new ZdImpl(inf, -pi / 2),
                oneNegInf().ln(), 10e-12);
        TestUtils.assertEquals(infZero(), infOne().ln(), 10e-12);
        TestUtils.assertEquals(new ZdImpl(inf, pi),
                negInfOne().ln(), 10e-12);
        TestUtils.assertEquals(new ZdImpl(inf, pi / 4),
                infInf().ln(), 10e-12);
        TestUtils.assertEquals(new ZdImpl(inf, -pi / 4),
                infNegInf().ln(), 10e-12);
        TestUtils.assertEquals(new ZdImpl(inf, 3d * pi / 4),
                negInfInf().ln(), 10e-12);
        TestUtils.assertEquals(new ZdImpl(inf, - 3d * pi / 4),
                negInfNegInf().ln(), 10e-12);
    }

    @Test
    public void testLogZero() {
        TestUtils.assertSame(negInfZero(), Zd.Zero().ln());
    }

    @Test
    public void testPow() {
        Zd x = new ZdImpl(3, 4);
        Zd y = new ZdImpl(5, 6);
        Zd expected = new ZdImpl(-1.860893, 11.83677);
        TestUtils.assertEquals(expected, x.pow(y), 1.0e-5);
    }

    @Test
    public void testPowNaNBase() {
        Zd x = new ZdImpl(3, 4);
        Assert.assertTrue(Zd.NaN().pow(x).isNan());
    }

    @Test
    public void testPowNaNExponent() {
        Zd x = new ZdImpl(3, 4);
        Assert.assertTrue(x.pow(Zd.NaN()).isNan());
    }

   @Test
   public void testPowInf() {
       TestUtils.assertSame(Zd.NaN(), Zd.One().pow(oneInf()));
       TestUtils.assertSame(Zd.NaN(), Zd.One().pow(oneNegInf()));
       TestUtils.assertSame(Zd.NaN(), Zd.One().pow(infOne()));
       TestUtils.assertSame(Zd.NaN(), Zd.One().pow(infInf()));
       TestUtils.assertSame(Zd.NaN(), Zd.One().pow(infNegInf()));
       TestUtils.assertSame(Zd.NaN(), Zd.One().pow(negInfInf()));
       TestUtils.assertSame(Zd.NaN(), Zd.One().pow(negInfNegInf()));
       TestUtils.assertSame(Zd.NaN(), infOne().pow(Zd.One()));
       TestUtils.assertSame(Zd.NaN(), negInfOne().pow(Zd.One()));
       TestUtils.assertSame(Zd.NaN(), infInf().pow(Zd.One()));
       TestUtils.assertSame(Zd.NaN(), infNegInf().pow(Zd.One()));
       TestUtils.assertSame(Zd.NaN(), negInfInf().pow(Zd.One()));
       TestUtils.assertSame(Zd.NaN(), negInfNegInf().pow(Zd.One()));
       TestUtils.assertSame(Zd.NaN(), negInfNegInf().pow(infNegInf()));
       TestUtils.assertSame(Zd.NaN(), negInfNegInf().pow(negInfNegInf()));
       TestUtils.assertSame(Zd.NaN(), negInfNegInf().pow(infInf()));
       TestUtils.assertSame(Zd.NaN(), infInf().pow(infNegInf()));
       TestUtils.assertSame(Zd.NaN(), infInf().pow(negInfNegInf()));
       TestUtils.assertSame(Zd.NaN(), infInf().pow(infInf()));
       TestUtils.assertSame(Zd.NaN(), infNegInf().pow(infNegInf()));
       TestUtils.assertSame(Zd.NaN(), infNegInf().pow(negInfNegInf()));
       TestUtils.assertSame(Zd.NaN(), infNegInf().pow(infInf()));
   }

   @Test
   public void testPowZero() {
       TestUtils.assertSame(Zd.NaN(),
               Zd.Zero().pow(Zd.One()));
       TestUtils.assertSame(Zd.NaN(),
               Zd.Zero().pow(Zd.Zero()));
       TestUtils.assertSame(Zd.NaN(),
               Zd.Zero().pow(Zd.I()));
       TestUtils.assertEquals(Zd.One(),
               Zd.One().pow(Zd.Zero()), 10e-12);
       TestUtils.assertEquals(Zd.One(),
               Zd.I().pow(Zd.Zero()), 10e-12);
       TestUtils.assertEquals(Zd.One(),
               new ZdImpl(-1, 3).pow(Zd.Zero()), 10e-12);
   }

    @Test
    public void testScalarPow() {
        Zd x = new ZdImpl(3, 4);
        double yDouble = 5.0;
        Zd yZdImpl = new ZdImpl(yDouble);
        Assert.assertEquals(x.pow(yZdImpl), x.pow(yDouble));
    }

    @Test
    public void testScalarPowNaNBase() {
        Zd x = Zd.NaN();
        double yDouble = 5.0;
        Zd yZdImpl = new ZdImpl(yDouble);
        Assert.assertEquals(x.pow(yZdImpl), x.pow(yDouble));
    }

    @Test
    public void testScalarPowNaNExponent() {
        Zd x = new ZdImpl(3, 4);
        double yDouble = Double.NaN;
        Zd yZdImpl = new ZdImpl(yDouble);
        Assert.assertEquals(x.pow(yZdImpl), x.pow(yDouble));
    }

   @Test
   public void testScalarPowInf() {
       TestUtils.assertSame(Zd.NaN(), Zd.One().pow(Double.POSITIVE_INFINITY));
       TestUtils.assertSame(Zd.NaN(), Zd.One().pow(Double.NEGATIVE_INFINITY));
       TestUtils.assertSame(Zd.NaN(), infOne().pow(1.0));
       TestUtils.assertSame(Zd.NaN(), negInfOne().pow(1.0));
       TestUtils.assertSame(Zd.NaN(), infInf().pow(1.0));
       TestUtils.assertSame(Zd.NaN(), infNegInf().pow(1.0));
       TestUtils.assertSame(Zd.NaN(), negInfInf().pow(10));
       TestUtils.assertSame(Zd.NaN(), negInfNegInf().pow(1.0));
       TestUtils.assertSame(Zd.NaN(), negInfNegInf().pow(Double.POSITIVE_INFINITY));
       TestUtils.assertSame(Zd.NaN(), negInfNegInf().pow(Double.POSITIVE_INFINITY));
       TestUtils.assertSame(Zd.NaN(), infInf().pow(Double.POSITIVE_INFINITY));
       TestUtils.assertSame(Zd.NaN(), infInf().pow(Double.NEGATIVE_INFINITY));
       TestUtils.assertSame(Zd.NaN(), infNegInf().pow(Double.NEGATIVE_INFINITY));
       TestUtils.assertSame(Zd.NaN(), infNegInf().pow(Double.POSITIVE_INFINITY));
   }

   @Test
   public void testScalarPowZero() {
       TestUtils.assertSame(Zd.NaN(), Zd.Zero().pow(1.0));
       TestUtils.assertSame(Zd.NaN(), Zd.Zero().pow(0.0));
       TestUtils.assertEquals(Zd.One(), Zd.One().pow(0.0), 10e-12);
       TestUtils.assertEquals(Zd.One(), Zd.I().pow(0.0), 10e-12);
       TestUtils.assertEquals(Zd.One(), new ZdImpl(-1, 3).pow(0.0), 10e-12);
   }

    @Test(expected=NullPointerException.class)
    public void testpowNull() {
        Zd.One().pow(null);
    }

    @Test
    public void testEqualsIssue() {
        Assert.assertEquals(new ZdImpl(0,-1), new ZdImpl(0,1).mul(new ZdImpl(-1,0)));
    }

    /**
     * Test standard values
     */
    @Test
    public void testArg() {
        Zd z = new ZdImpl(1, 0);
        Assert.assertEquals(0.0, z.arg(), 1.0e-12);

        z = new ZdImpl(1, 1);
        Assert.assertEquals(Math.PI/4, z.arg(), 1.0e-12);

        z = new ZdImpl(0, 1);
        Assert.assertEquals(Math.PI/2, z.arg(), 1.0e-12);

        z = new ZdImpl(-1, 1);
        Assert.assertEquals(3 * Math.PI/4, z.arg(), 1.0e-12);

        z = new ZdImpl(-1, 0);
        Assert.assertEquals(Math.PI, z.arg(), 1.0e-12);

        z = new ZdImpl(-1, -1);
        Assert.assertEquals(-3 * Math.PI/4, z.arg(), 1.0e-12);

        z = new ZdImpl(0, -1);
        Assert.assertEquals(-Math.PI/2, z.arg(), 1.0e-12);

        z = new ZdImpl(1, -1);
        Assert.assertEquals(-Math.PI/4, z.arg(), 1.0e-12);
    }

    /**
     * Verify atan2-style handling of infinite parts
     */
    @Test
    public void testArgInf() {
        Assert.assertEquals(Math.PI/4, infInf().arg(), 1.0e-12);
        Assert.assertEquals(Math.PI/2, oneInf().arg(), 1.0e-12);
        Assert.assertEquals(0.0, infOne().arg(), 1.0e-12);
        Assert.assertEquals(Math.PI/2, zeroInf().arg(), 1.0e-12);
        Assert.assertEquals(0.0, infZero().arg(), 1.0e-12);
        Assert.assertEquals(Math.PI, negInfOne().arg(), 1.0e-12);
        Assert.assertEquals(-3.0*Math.PI/4, negInfNegInf().arg(), 1.0e-12);
        Assert.assertEquals(-Math.PI/2, oneNegInf().arg(), 1.0e-12);
    }

    /**
     * Verify that either part NaN results in NaN
     */
    @Test
    public void testArgNaN() {
        Assert.assertTrue(Double.isNaN(nanZero().arg()));
        Assert.assertTrue(Double.isNaN(zeroNaN().arg()));
        Assert.assertTrue(Double.isNaN(Zd.NaN().arg()));
    }
}
