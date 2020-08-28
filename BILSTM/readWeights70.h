#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <math.h>
#include <stdlib.h>
using namespace std;
void readWeights(double *X,double *W_i,double *W_f,double *W_c,double *W_o,double *U_i,double *U_f,double *U_c,double *U_o,double *b_i,double *b_f,double *b_c,double *b_o,double *w,double *b)
{
 string line;
        ifstream f11("weights/W_i_bilstm_70.txt");
        if(f11.is_open())
        {
                int c1=0;
                while(getline(f11,line))
                {
                        string s_tmp="";
                        double i_tmp;

                        vector<double> numbers;
                        for(vector<double>::size_type i = 0, len = line.size(); i < len; i++){
                                if( line[i] == ',' ){
                                        if(s_tmp.size() > 0){
                                                i_tmp = std::stod(s_tmp);
                                                numbers.push_back(i_tmp);
                                        }
                                        s_tmp = "";
                                }
                                else if( i == len-1){
                                        s_tmp += line[i];
                                        i_tmp = std::stod(s_tmp);
                                        numbers.push_back(i_tmp);

                                }
                                else {
                                        s_tmp +=line[i];
                                }
                        }
                        int c2=0;
                        for (auto it = numbers.begin(); it != numbers.end(); it++){
                                *(W_i+c1*3+c2)=*it;
                                c2++;}
                        c1++;
                }
        }
        ifstream f12("weights/W_f_bilstm_70.txt");
        if(f12.is_open())
        {
                int c1=0;
                while(getline(f12,line))
                {
                        string s_tmp="";
                        double i_tmp;
                        vector<double> numbers;
                        for(vector<double>::size_type i = 0, len = line.size(); i < len; i++){
                                if( line[i] == ',' ){
                                        if(s_tmp.size() > 0){
                                                i_tmp = std::stod(s_tmp);
                                                numbers.push_back(i_tmp);
                                        }
                                        s_tmp = "";
                                }
                                else if( i == len-1){
                                        s_tmp += line[i];
                                        i_tmp = std::stod(s_tmp);
                                        numbers.push_back(i_tmp);

                                }
                                else {
                                        s_tmp +=line[i];
                                }
                        }
                        int c2=0;
                        for(auto it = numbers.begin(); it != numbers.end(); it++){
                                *(W_f+c1*3+c2)=*it;
                                c2++;}
                        c1++;
                }
        }
        ifstream f13("weights/W_c_bilstm_70.txt");
        if(f13.is_open())
        {
                int c1=0;
                while(getline(f13,line))
                {
                        string s_tmp="";
                        double i_tmp;
                        
                        vector<double> numbers;
                        for(vector<double>::size_type i = 0, len = line.size(); i < len; i++){
                                if( line[i] == ',' ){
                                        if(s_tmp.size() > 0){
                                                i_tmp = std::stod(s_tmp);
                                                numbers.push_back(i_tmp);
                                        }
                                        s_tmp = "";
                                }
                                else if( i == len-1){
                                        s_tmp += line[i];
                                        i_tmp = std::stod(s_tmp);
                                        numbers.push_back(i_tmp);

                                }
                                else {
                                        s_tmp +=line[i];
                                }
                        }
                        int c2=0;
                        for (auto it = numbers.begin(); it != numbers.end(); it++){
                                *(W_c+c1*3+c2)=*it;
                                c2++;}
                        c1++;
                }
        }
        ifstream f14("weights/W_o_bilstm_70.txt");
        if(f14.is_open())
        {
                int c1=0;
                while(getline(f14,line))
                {
                        string s_tmp="";
                        double i_tmp;
                        
                        vector<double> numbers;
                        for(vector<double>::size_type i = 0, len = line.size(); i < len; i++){
                                if( line[i] == ',' ){
                                        if(s_tmp.size() > 0){
                                                i_tmp = std::stod(s_tmp);
                                                numbers.push_back(i_tmp);
                                        }
                                        s_tmp = "";
                                }
                                else if( i == len-1){
                                        s_tmp += line[i];
                                        i_tmp = std::stod(s_tmp);
                                        numbers.push_back(i_tmp);

                                }
                                else {
                                        s_tmp +=line[i];
                                }
                        }
                        int c2=0;
                        for (auto it = numbers.begin(); it != numbers.end(); it++){
                                *(W_o+c1*3+c2)=*it;
                                c2++;}
                        c1++;
                }
        }
	 ifstream f21("weights/U_i_bilstm_70.txt");
        if(f21.is_open())
        {
                int c1=0;
                while(getline(f21,line))
                {
                        string s_tmp="";
                        double i_tmp;
                        
                        vector<double> numbers;
                        for(vector<double>::size_type i = 0, len = line.size(); i < len; i++){
                                if( line[i] == ',' ){
                                        if(s_tmp.size() > 0){
                                                i_tmp = std::stod(s_tmp);
                                                numbers.push_back(i_tmp);
                                        }
                                        s_tmp = "";
                                }
                                else if( i == len-1){
                                        s_tmp += line[i];
                                        i_tmp = std::stod(s_tmp);
                                        numbers.push_back(i_tmp);

                                }
                                else {
                                        s_tmp +=line[i];
                                }
                        }
                        int c2=0;
                        for (auto it = numbers.begin(); it != numbers.end(); it++){
                                *(U_i+c1*70+c2)=*it;
                                c2++;}
                        c1++;
                }
        }
        ifstream f22("weights/U_f_bilstm_70.txt");
        if(f22.is_open())
        {
                int c1=0;
                while(getline(f22,line))
                {
                        string s_tmp="";
                        double i_tmp;
                        
                        vector<double> numbers;
                        for(vector<double>::size_type i = 0, len = line.size(); i < len; i++){
                                if( line[i] == ',' ){
                                        if(s_tmp.size() > 0){
                                                i_tmp = std::stod(s_tmp);
                                                numbers.push_back(i_tmp);
                                        }
                                        s_tmp = "";
                                }
                                else if( i == len-1){
                                        s_tmp += line[i];
                                        i_tmp = std::stod(s_tmp);
                                        numbers.push_back(i_tmp);

                                }
                                else {
                                        s_tmp +=line[i];
                                }
                        }
                        int c2=0;
                        for (auto it = numbers.begin(); it != numbers.end(); it++){
                                *(U_f+c1*70+c2)=*it;
                                c2++;}
                        c1++;
                }
        }
        ifstream f23("weights/U_c_bilstm_70.txt");
        if(f23.is_open())
        {
                int c1=0;
                while(getline(f23,line))
                {
                        string s_tmp="";
                        double i_tmp;
                        
                        vector<double> numbers;
                        for(vector<double>::size_type i = 0, len = line.size(); i < len; i++){
                                if( line[i] == ',' ){
                                        if(s_tmp.size() > 0){
                                                i_tmp = std::stod(s_tmp);
                                                numbers.push_back(i_tmp);
                                        }
                                        s_tmp = "";
                                }
                                else if( i == len-1){
                                        s_tmp += line[i];
                                        i_tmp = std::stod(s_tmp);
                                        numbers.push_back(i_tmp);

                                }
                                else {
                                        s_tmp +=line[i];
                                }
                        }
                        int c2=0;
                        for (auto it = numbers.begin(); it != numbers.end(); it++){
                                *(U_c+c1*70+c2)=*it;
                                c2++;}
                        c1++;
                }
        }
        ifstream f24("weights/U_o_bilstm_70.txt");
        if(f24.is_open())
        {
                int c1=0;
                while(getline(f24,line))
                {
                        string s_tmp="";
                        double i_tmp;
                        
                        vector<double> numbers;
                        for(vector<double>::size_type i = 0, len = line.size(); i < len; i++){
                                if( line[i] == ',' ){
                                        if(s_tmp.size() > 0){
                                                i_tmp = std::stod(s_tmp);
                                                numbers.push_back(i_tmp);
                                        }
                                        s_tmp = "";
                                }
                                else if( i == len-1){
                                        s_tmp += line[i];
                                        i_tmp = std::stod(s_tmp);
                                        numbers.push_back(i_tmp);

                                }
                                else {
                                        s_tmp +=line[i];
                                }
                        }
                        int c2=0;
                        for (auto it = numbers.begin(); it != numbers.end(); it++){
                                *(U_o+c1*70+c2)=*it;
                                c2++;}
                        c1++;
                }
        }
        ifstream f31("weights/b_i_bilstm_70.txt");
        if(f31.is_open())
        {
                int c1=0;
                while(getline(f31,line))
                {
                        string s_tmp="";
                        double i_tmp;
                        
                        vector<double> numbers;
                        for(vector<double>::size_type i = 0, len = line.size(); i < len; i++){
                                if( line[i] == ',' ){
                                        if(s_tmp.size() > 0){
                                                i_tmp = std::stod(s_tmp);
                                                numbers.push_back(i_tmp);
                                        }
                                        s_tmp = "";
                                }
                                else if( i == len-1){
                                        s_tmp += line[i];
                                        i_tmp = std::stod(s_tmp);
                                        numbers.push_back(i_tmp);

                                }
                                else {
                                        s_tmp +=line[i];
                                }
                        }
                        for (auto it = numbers.begin(); it != numbers.end(); it++)
			{
                                *(b_i+c1)=*it;
			}
			
                        c1++;
                }
        }
        ifstream f32("weights/b_f_bilstm_70.txt");
        if(f32.is_open())
        {
                int c1=0;
                while(getline(f32,line))
                {
                        string s_tmp="";
                        double i_tmp;
                        
                        vector<double> numbers;
                        for(vector<double>::size_type i = 0, len = line.size(); i < len; i++){
                                if( line[i] == ',' ){
                                        if(s_tmp.size() > 0){
                                                i_tmp = std::stod(s_tmp);
                                                numbers.push_back(i_tmp);
                                        }
                                        s_tmp = "";
                                }
                                else if( i == len-1){
                                        s_tmp += line[i];
                                        i_tmp = std::stod(s_tmp);
                                        numbers.push_back(i_tmp);

                                }
                                else {
                                        s_tmp +=line[i];
                                }
                        }
                        for (auto it = numbers.begin(); it != numbers.end(); it++)
                                *(b_f+c1)=*it;
                        c1++;
                }
        }
        ifstream f33("weights/b_c_bilstm_70.txt");
        if(f33.is_open())
        {
                int c1=0;
                while(getline(f33,line))
                {
                        string s_tmp="";
                        double i_tmp;
                        
                        vector<double> numbers;
                        for(vector<double>::size_type i = 0, len = line.size(); i < len; i++){
                                if( line[i] == ',' ){
                                        if(s_tmp.size() > 0){
                                                i_tmp = std::stod(s_tmp);
                                                numbers.push_back(i_tmp);
                                        }
                                        s_tmp = "";
                                }
                                else if( i == len-1){
                                        s_tmp += line[i];
                                        i_tmp = std::stod(s_tmp);
                                        numbers.push_back(i_tmp);

                                }
                                else {
                                        s_tmp +=line[i];
                                }
                        }
                        for (auto it = numbers.begin(); it != numbers.end(); it++)
                                *(b_c+c1)=*it;
                        c1++;
                }
        }
        ifstream f34("weights/b_o_bilstm_70.txt");
        if(f34.is_open())
        {
                int c1=0;
                while(getline(f34,line))
                {
                        string s_tmp="";
                        double i_tmp;
                        
                        vector<double> numbers;
                        for(vector<double>::size_type i = 0, len = line.size(); i < len; i++){
                                if( line[i] == ',' ){
                                        if(s_tmp.size() > 0){
                                                i_tmp = std::stod(s_tmp);
                                                numbers.push_back(i_tmp);
                                        }
                                        s_tmp = "";
                                }
                                else if( i == len-1){
                                        s_tmp += line[i];
                                        i_tmp = std::stod(s_tmp);
                                        numbers.push_back(i_tmp);

                                }
                                else {
                                        s_tmp +=line[i];
                                }
                        }
                        for (auto it = numbers.begin(); it != numbers.end(); it++)
                                *(b_o+c1)=*it;
                        c1++;
                }
        }
	ifstream file("weights/dense_bilstm_70.txt");
        if(file.is_open())
        {
                int c1=0;
                while(getline(file,line))
                {
                        string s_tmp="";
                        double i_tmp;
                        
                        vector<double> numbers;
                        for(vector<double>::size_type i = 0, len = line.size(); i < len; i++){
                                if( line[i] == ',' ){
                                        if(s_tmp.size() > 0){
                                                i_tmp = std::stod(s_tmp);
                                                numbers.push_back(i_tmp);
                                        }
                                        s_tmp = "";
                                }
                                else if( i == len-1){
                                        s_tmp += line[i];
                                        i_tmp = std::stod(s_tmp);
                                        numbers.push_back(i_tmp);

                                }
                                else {
                                        s_tmp +=line[i];
                                }
                        }
                        int c2=0;
                        if(c1<12)
                        {
                                for (auto it = numbers.begin(); it != numbers.end(); it++)
                                        *(b+c1)=*it;
                        }
                        else
                        {
                                for (auto it = numbers.begin(); it != numbers.end(); it++){
                                        *(w+(c1-12)*12+c2)=*it;
                                        c2++;}
                        }
                        c1++;
                }
        }
        ifstream file1("final_aruba1.txt");
        if(file1.is_open())
        {
                int c1=0;
                while(getline(file1,line))
                {
                        int c2=0;
                        string word = "";
                        for (auto x : line)
                        {
                                if (x == ' ')
                                {
                                        if(c2<3)
                                        {
                                                *(X+c1*3+c2)=std::stod(word);
                                                c2++;
                                        }
                                        word = "";
                                }
                                else
                                {
                                        word = word + x;
                                }
                        }
                        if(c2<3)
                        {
                                *(X+c1*3+c2)=std::stod(word);
                                c2++;
                        }
                        c1++;
                }
        }

}
