---
layout: post
title: scala方法整理
category: SPARK
tags: spark
description: 主要收集scala方法
---


```scala

case 自动生成构造函数，trait没有构造函数可以有变量

//int ret = openURL(*env, "test", "rtsp://192.168.42.1/live");
	int ret = openURL(*env, "test", "rtsp://192.168.80.95:8554/");

TextAsset movieTex = Resources.Load("Image/a1") as TextAsset;

        Texture2D texCopy = new Texture2D(2, 2);
        texCopy.LoadImage(movieTex.bytes);
        texCopy.Apply();

        tex = new Texture2D(texCopy.width/2, texCopy.height, texCopy.format, true);
        tex.SetPixels(texCopy.GetPixels(0, 0, texCopy.width/2, texCopy.height));
        tex.Apply();



 if(null == GameObject.FindGameObjectWithTag("7_14x")) {
                      
                Debug.Log("Meta1_Shake: " + "888888888");
                Instantiate(Resources.Load("7_14x"));
            } else {
                Debug.Log("Meta1_Shake: " + "77777777");
            }


 /*
19     * 写法二，省略不是Unit类型的返回值类型；如果没有写返回值，则根据等号后面的内容进行类型推演
20     */
21    def test(x:Int)={
22       x
23    }
24     
25    /*
26     * 写法三，省略等号，返回Unit
27     */
28    def returnVoid(){
29      println("返回 void")
30    }   
31    def returnUnit():Unit={
32      println("另外一种方法返回 void")
33    }

def this(fn : String, ln : String)  { 
11     this(fn, ln, null)
12   }

// implicit approach
val add = (x: Int, y: Int) => { x + y }
val add = (x: Int, y: Int) => x + y

// explicit approach
val add: (Int, Int) => Int = (x,y) => { x + y }
val add: (Int, Int) => Int = (x,y) => x + y

匿名函数 授予方法
// implicit approach
val add = (x: Int, y: Int) => { x + y }
val add = (x: Int, y: Int) => x + y

// explicit approach
val add: (Int, Int) => Int = (x,y) => { x + y }
val add: (Int, Int) => Int = (x,y) => x + y

type IntPairPred = (Int, Int) => Boolean
val gt: IntPairPred = _ > _
val ge: IntPairPred = _ >= _
val lt: IntPairPred = _ < _
val le: IntPairPred = _ <= _
val eq: IntPairPred = _ == _

def modMethod(i: Int) = i % 2 == 0
def modMethod(i: Int) = { i % 2 == 0 }
def modMethod(i: Int): Boolean = i % 2 == 0
def modMethod(i: Int): Boolean = { i % 2 == 0 }

val list = List.range(1, 10)
list.filter(modMethod)


type IntPairPred = (Int, Int) => Boolean
def sizeConstraint(pred: IntPairPred, n: Int, email: Email) = pred(email.text.size, n)
val constr20: (IntPairPred, Email) => Boolean = sizeConstraint(_: IntPairPred, 20, _: Email)

val sizeConstraintFn: (IntPairPred, Int, Email) => Boolean = sizeConstraint _

def sizeConstraint(pred: IntPairPred)(n: Int)(email: Email): Boolean =
  pred(email.text.size, n)

val sizeConstraintFn: IntPairPred => Int => Email => Boolean = sizeConstraint _

val sum: (Int, Int) => Int = _ + _
val sumCurried: Int => Int => Int = sum.curried
http://danielwestheide.com/blog/2013/01/30/the-neophytes-guide-to-scala-part-11-currying-and-partially-applied-functions.html  Currying and partial function


 def add1(x: Int)(y: Int) = x + y
  
  def add2(x: Int) = (y: Int) => x + y

//默认参数
println("\nStep 3: How to add default values to function parameters")
def calculateDonutCost(donutName: String, quantity: Int, couponCode: String = "NO CODE"): Double = {
  println(s"Calculating cost for $donutName, quantity = $quantity, couponCode = $couponCode")
// make some calculations ...
2.50 * quantity
}
val totalCostWithDiscount = calculateDonutCost("Glazed Donut", 4, "COUPON_12345")
val totalCostWithoutDiscount = calculateDonutCost("Glazed Donut", 4)

可选
println("\nStep 3: How to assign a default value to an Option parameter")
def calculateDonutCostWithDefaultOptionValue(donutName: String, quantity: Int, couponCode: Option[String] = None): Double = {
  println(s"Calculating cost for $donutName, quantity = $quantity")

  couponCode match{
    case Some(coupon) =>
      val discount = 0.1 // Let's simulate a 10% discount
      val totalCost = 2.50 * quantity * (1 - discount)
      totalCost

    case _ => 2.50 * quantity
  }
}

println(s"""Total cost = ${calculateDonutCostWithDefaultOptionValue("Glazed Donut", 5)}""")
println(s"""Total cost with discount = ${calculateDonutCostWithDefaultOptionValue("Glazed Donut", 5, Some("COUPON_1234"))}""")

Calculating cost for Glazed Donut, quantity = 5
Total cost = 12.5
Calculating cost for Glazed Donut, quantity = 5
Total cost with discount = 11.25

潜在函数 implicit，管着两个参数
println("\nStep 4: How to define a function which takes multiple implicit parameters")
def totalCost2(donutType: String, quantity: Int)(implicit discount: Double, storeName: String): Double = {
  println(s"[$storeName] Calculating the price for $quantity $donutType")
  val totalCost = 2.50 * quantity * (1 - discount)
  totalCost
}
implicit val discount: Double = 0.1
implicit val storeName: String = "Tasty Donut Store"
println(s"""Total cost with discount of 5 Glazed Donuts = ${totalCost2("Glazed Donut", 5)}""")就不需要传参数
println(s"""Total cost with discount of 5 Glazed Donuts, manually passed-through = ${totalCost2("Glazed Donut", 5)(0.1, "Scala Donut Store")}""")

多参数printReport("Chocolate Donut")
def printReport(names: String*) {
println(s"""Donut Report = ${names.mkString(", ")}""")
}

printReport("Chocolate Donut")
val seqDonuts: Seq[String] = Seq("Chocolate Donut", "Plain Donut")
printReport(seqDonuts: _*) 加星号

函数作为参数
def totalCostWithDiscountFunctionParameter(donutType: String)(quantity: Int)(f: Double => Double): Double = {
  println(s"Calculating total cost for $quantity $donutType")
  val totalCost = 2.50 * quantity
  f(totalCost)
}
 val totalCostOf5Donuts = totalCostWithDiscountFunctionParameter("Glazed Donut")(5){totalCost =>
  val discount = 2 // assume you fetch discount from database
  totalCost - discount
}
println(s"Total cost of 5 Glazed Donuts with anonymous discount function = $totalCostOf5Donuts")

def applyDiscount(totalCost: Double): Double = {
  val discount = 2 // assume you fetch discount from database
  totalCost - discount
}
println(s"Total cost of 5 Glazed Donuts with discount function = ${totalCostWithDiscountFunctionParameter("Glazed Donut")(5)(applyDiscount(_))}")


回调函数
println("\nStep 4: How to define a function Function with an Option callback")
def printReportWithOptionCallback(sendEmailCallback: Option[() => Unit] = None) {
  println("Printing report ... started")
  // look up some data in database and create a report
  println("Printing report ... finished")
  sendEmailCallback.map(callback => callback())
}
printReportWithOptionCallback(Some(() =>
  println("Sending email wrapped in Some() ... finished")
))


传递函数参数引用
def totalCostWithDiscountFunctionParameter(donutType: String)(quantity: Int)(f: Double => Double): Double = {
  println(s"Calculating total cost for $quantity $donutType")
  val totalCost = 2.50 * quantity
  f(totalCost)
}
def applyDiscount(totalCost: Double): Double = {
  val discount = 2 // assume you fetch discount from database
  totalCost - discount
}
通过函数名调用 要用下划线
println(s"Total cost of 5 Glazed Donuts with discount def function = ${totalCostWithDiscountFunctionParameter("Glazed Donut")(5)(applyDiscount(_))}")

val applyDiscountValueFunction = (totalCost: Double) => {
  val discount = 2 // assume you fetch discount from database
  totalCost - discount
}
通过变量引用不需要下划线
println(s"Total cost of 5 Glazed Donuts with discount val function = ${totalCostWithDiscountFunctionParameter("Glazed Donut")(5)(applyDiscountValueFunction)}")

and then的使用
val applyDiscountValFunction = (amount: Double) => {
  println("Apply discount function")
  val discount = 2 // fetch discount from database
  amount - discount
}
val applyTaxValFunction = (amount: Double) => {
  println("Apply tax function")
  val tax = 1 // fetch tax from database
  amount + tax
}
println(s"Total cost of 5 donuts = ${ (applyDiscountValFunction andThen applyTaxValFunction)(totalCost) }")

Ordering using andThen: f(x) andThen g(x) = g(f(x))
Ordering using compose: f(x) compose g(x) = f(g(x))


递归
@annotation.tailrec
def search(donutName: String, donuts: Array[String], index: Int): Option[Boolean] = {
  if(donuts.length == index) {
    None
  } else if(donuts(index) == donutName) {
    Some(true)
  } else {
    val nextIndex = index + 1
    search(donutName, donuts, nextIndex)
  }
}
val arrayDonuts: Array[String] = Array("Vanilla Donut", "Strawberry Donut", "Plain Donut", "Glazed Donut")
val found = search("Glazed Donut", arrayDonuts, 0)
println(s"Find Glazed Donut = $found")

val notFound = search("Chocolate Donut", arrayDonuts, 0)
println(s"Find Chocolate Donut = $notFound")

导入包递归package scala.util.control.TailCalls._ 
ef tailSearch(donutName: String, donuts: Array[String], index: Int): TailRec[Option[Boolean]] = {
  if(donuts.length == index) {
    done(None) // NOTE: done is imported from scala.util.control.TailCalls._
  } else if(donuts(index) == donutName) {
    done(Some(true))
  } else {
    val nextIndex = index + 1
    tailcall(tailSearch(donutName, donuts, nextIndex)) // NOTE: tailcall is imported from  scala.util.control.TailCalls._
  }
}

println("\nStep 4: How to call tail recursive function using scala.util.control.TailCalls._")
val tailFound = tailcall(tailSearch("Glazed Donut", arrayDonuts, 0))
println(s"Find Glazed Donut using TailCall = ${tailFound.result}") // NOTE: our returned value is wrapped so we need to get it by calling result

val tailNotFound = tailcall(tailSearch("Chocolate Donut", arrayDonuts, 0))
println(s"Find Chocolate Donut using TailCall = ${tailNotFound.result}")

单独匹配 将模式匹配单独分开 orElse
val donut = "Glazed Donut"
val tasteLevel = donut match {
  case "Glazed Donut" | "Strawberry Donut" => "Very tasty"
  case "Plain Donut" => "Tasty"
  case _ => "Tasty"
}
println(s"Taste level of $donut = $tasteLevel")

val isVeryTasty: PartialFunction[String, String] = {
  case "Glazed Donut" | "Strawberry Donut" => "Very Tasty"
}
println(s"Calling partial function isVeryTasty = ${isVeryTasty("Glazed Donut")}")
val isTasty: PartialFunction[String, String] = {
  case "Plain Donut" => "Tasty"
}

val unknownTaste: PartialFunction[String, String] = {
  case donut @ _ => s"Unknown taste for donut = $donut"
}

val donutTaste = isVeryTasty orElse isTasty orElse unknownTaste
println(donutTaste("Glazed Donut"))
println(donutTaste("Plain Donut"))
println(donutTaste("Chocolate Donut"))

枚举
object Donut extends Enumeration {
  type Donut = Value

  val Glazed      = Value("Glazed")
  val Strawberry  = Value("Strawberry")
  val Plain       = Value("Plain")
  val Vanilla     = Value("Vanilla")
}
println(s"Vanilla Donut string value = ${Donut.Vanilla}")
println(s"Vanilla Donut's id = ${Donut.Vanilla.id}")
println(s"Donut types = ${Donut.values}")

object DonutTaste extends Enumeration{
  type DonutTaste = Value

  val Tasty       = Value(0, "Tasty")
  val VeryTasty   = Value(1, "Very Tasty")
  val Ok          = Value(-1, "Ok")
}

println(s"Donut taste values = ${DonutTaste.values}")
println(s"Donut taste of OK id = ${DonutTaste.Ok.id}")


https://www.supergloo.com/fieldnotes/apache-spark-examples-of-transformations/  rdd 操作

scala> sc.parallelize(List(1,2,3)).flatMap(x=>List(x,x,x)).collect
res200: Array[Int] = Array(1, 1, 1, 2, 2, 2, 3, 3, 3)
 
scala> sc.parallelize(List(1,2,3)).map(x=>List(x,x,x)).collect
res201: Array[List[Int]] = Array(List(1, 1, 1), List(2, 2, 2), List(3, 3, 3))


 
scala> sc.parallelize(List(1,2,3)).flatMap(x=>List(x,x,x))
res202: org.apache.spark.rdd.RDD[Int] = FlatMappedRDD[373] at flatMap at <console>:13
 
scala> sc.parallelize(List(1,2,3)).map(x=>List(x,x,x))
res203: org.apache.spark.rdd.RDD[List[Int]] = MappedRDD[375] at map at <console>:13


// from laptop
scala> val parallel = sc.parallelize(1 to 9, 3)
parallel: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[450] at parallelize at <console>:12
 
scala> parallel.mapPartitions( x => List(x.next).iterator).collect
res383: Array[Int] = Array(1, 4, 7)
 
// compare to the same, but with default parallelize
scala> val parallel = sc.parallelize(1 to 9)
parallel: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[452] at parallelize at <console>:12
 
scala> parallel.mapPartitions( x => List(x.next).iterator).collect
res384: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7, 8)

mapPartitionsWithIndex(func)

scala> val parallel = sc.parallelize(1 to 9)
parallel: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[455] at parallelize at <console>:12
 
scala> parallel.mapPartitionsWithIndex( (index: Int, it: Iterator[Int]) => it.toList.map(x => index + ", "+x).iterator).collect
res389: Array[String] = Array(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 7, 9)

scala> val parallel = sc.parallelize(1 to 9, 3)
parallel: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[457] at parallelize at <console>:12
 
scala> parallel.mapPartitionsWithIndex( (index: Int, it: Iterator[Int]) => it.toList.map(x => index + ", "+x).iterator).collect
res390: Array[String] = Array(0, 1, 0, 2, 0, 3, 1, 4, 1, 5, 1, 6, 2, 7, 2, 8, 2, 9)


union(a different rdd)

scala> val parallel = sc.parallelize(1 to 9)
parallel: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[477] at parallelize at <console>:12
 
scala> val par2 = sc.parallelize(5 to 15)
par2: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[478] at parallelize at <console>:12
 
scala> parallel.union(par2).collect
res408: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

intersection(a different rdd)

scala> val parallel = sc.parallelize(1 to 9)
parallel: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[477] at parallelize at <console>:12
 
scala> val par2 = sc.parallelize(5 to 15)
par2: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[478] at parallelize at <console>:12
 
scala> parallel.intersection(par2).collect
res409: Array[Int] = Array(8, 9, 5, 6, 7)

distinct([numTasks])

scala> val parallel = sc.parallelize(1 to 9)
parallel: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[477] at parallelize at <console>:12
 
scala> val par2 = sc.parallelize(5 to 15)
par2: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[478] at parallelize at <console>:12
 
scala> parallel.union(par2).distinct.collect
res412: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

join(otherDataset, [numTasks])

scala> val names1 = sc.parallelize(List("abe", "abby", "apple")).map(a => (a, 1))
names1: org.apache.spark.rdd.RDD[(String, Int)] = MappedRDD[1441] at map at <console>:14
 
scala> val names2 = sc.parallelize(List("apple", "beatty", "beatrice")).map(a => (a, 1))
names2: org.apache.spark.rdd.RDD[(String, Int)] = MappedRDD[1443] at map at <console>:14
 
scala> names1.join(names2).collect
res735: Array[(String, (Int, Int))] = Array((apple,(1,1)))
 
scala> names1.leftOuterJoin(names2).collect
res736: Array[(String, (Int, Option[Int]))] = Array((abby,(1,None)), (apple,(1,Some(1))), (abe,(1,None)))
 
scala> names1.rightOuterJoin(names2).collect
res737: Array[(String, (Option[Int], Int))] = Array((apple,(Some(1),1)), (beatty,(None,1)), (beatrice,(None,1)))

reduceByKey(func, [numTasks])
reduceByKey(func: (V, V) ⇒ V): RDD[(K, V)]

scala> val onlyInterestedIn = sc.textFile("baby_names.csv").map(line => line.split(",")).map(n => (n(1), n(4)))
onlyInterestedIn: org.apache.spark.rdd.RDD[(String, String)] = MappedRDD[27] at map at <console>:12
 
scala> onlyInterestedIn.saveAsTextFile("results.csv")


```

##spark 

https://www.supergloo.com/fieldnotes/intellij-scala-spark/  IntelliJ spark 安装和运行demo

http://www.beingsoftwareprofessional.com/2016/02/15/apache-spark-building-applications-with-maven-eclipse/

http://learningapachespark.blogspot.de/2015/03/12-how-to-run-spark-with-eclipse-and.html

https://github.com/H4ml3t/spark-scala-maven-boilerplate-project

https://www.slideshare.net/cloudera/top-5-mistakes-to-avoid-when-writing-apache-spark-applications?qid=842234e3-4696-4f34-a441-6891bedd90a1&v=&b=&from_search=10 计算core node
 
