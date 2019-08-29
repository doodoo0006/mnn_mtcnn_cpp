# mnn_mtcnn_cpp
mnn based mtcnn c++ realize.
目前安卓端调用mnn的demo是将应用层（如loadmodel， create session）逻辑写入了java层， 对于不熟悉java的同学开发不是很方便， 本文提供一个版本将应用层写到c++的demo演示。这样做的好处是核心算法逻辑和mnn实现都可以在c++层完成，java层只需要传图和显示结果即可，而且只需要一个jni接口交互， 避免了在java和c++对算法混合开发的过程。
