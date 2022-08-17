## About the parallelization of the ```URL GET``` requests

### TL;DR
Each request was made by a thread. The thread number was choosen to be 50.

### Detailed
This solution was heavily inspired by the code found on the link bellow

https://www.shanelynn.ie/using-python-threading-for-multiple-results-queue/

The ideia is basically to used threads to generate the requests (knowing that we don't have a real thread parallelism in python, but what seems to be one, given that all is executed concurrently). But then we encounter the basic problems of the parallelism universe: 

1. __Are we going to use a shared variable?__
   
    R: Yes, we are. But since the parallelism idea comes to the fact that there is a list of URLs where each request of one URL of it does not interfere on the request of another, the we can simply create a NEW array of the same size of that one, populate it with empty values (or strings or empty arrays) and then populate it on each request given that we pass an index of the URL on the first array to that new one. If it sounded confused, the idea is: the preprocessed data of the URL that occupies a given index on the first array will be at the same index of the second one. Just like that.

2. __How can we choose an adequate number of threads to solve this problem without overflowing our RAM memory or the native python's thread creation limit (the python has a quite obscure limit number of threads that it can create)?__

    R: This is purely arbitrary :) I just let the number used in the tutorial as it was. It seemed not so big and not so small.

3. __How each thread will know the number of URLs it can process requests (since some threads can finish faster than others and so on)?__

    R: For this we'll use the multithreading data structure ```Queue()``` from the ```queue``` library. We populate it if pairs (index, url_correspondent_that_index) of the original array. And on each thread, we simply verify if the queue is not empty. If that's the case, we can pop an element of it (that will be reflected to the other threads) and make a request with it. Since we have also the index of the original element on the original array, it is as described on item __1)__ above.

