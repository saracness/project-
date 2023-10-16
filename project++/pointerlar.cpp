#include <iostream>

using namespace std;
int yn(int* pr) {
	*pr = 45;                        //memory harcayarak x in deðerini deðiþtirip kopyalýyor
	return *pr;
}
                         //pointerlrda amaç vakit kaybetmeden senin isteðin üzerinde düzenleme yapar poinerdeki amaç milisaniyelerde deðer atamaktýr tüm satýra;
void n(int *p) {


	                                //harcamadan direkt olarak eriþerek deðiþtiriyor pointer üzerinden;
	*p = 35;
}
/*Mesela her milisaniyede bir etrafýnda hareket olduðunu anlamaya çalýþan bi dedektör içi
bu þekilde atadýðýnýz pointer o tüm satýrý her saniye kolayca yazar*/

void fon(int &x) {
	x *= 3;

	cout << x;

}


int main() {
	/*int x = 0;

	int* ptr;
	ptr = &x;


	cout << *ptr;
	cout << endl;
	*ptr = 2;
	cout << *ptr;
	cout << endl;
	*/
	

	

	int dizi[3][3]{ {12, 13, 14 },
	{12, 14, 15},
	{12, 11, 13},
	};

	for (int i = 0; i <= 2; i++) {
		for (int j = 0; j <= 2; j++) {
			cout << dizi[i][j];
			cout << " ";
		}
		cout << endl;
	}
	cout << endl;

	int arr[] = { 12,13,15 };
	int* p;
	p = arr;

	cout << *(p+1);
	cout << p[1];
	cout << endl;

	int x = 5;
	
	 fon(x);
	 cout << endl;
	 cout << "main icindeki x" << x;


	cout << endl;

	return 0;
}

