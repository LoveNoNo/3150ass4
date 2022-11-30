#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume; // Array Size: 1060 K; Element: unchar = 1 Byte = 8 bits
    /* 
      structure design
      4 KB = Bit-Vector: use to free space managment (first 4 KB)
        uchar volume[4096]
        set in write & delete
      32 KB = FCB: 1024 entries with size 32 Byte
        FCB entries volume[1024*32], for enevry [32] entry
        [0 ~ 19] 
        20 B for file name
        [20 ~ 21]; [22]  
        2 B for start block num (<32K); 1 B for block length (<32);
          block num from 0 to 32K, block 0 in File_Base_Address(4K + 32K) 
        [23]
        1 B for modify order：这是高位 ps：原本只给了一位是误理解了修改时间为修改次数，次数一位就够了
        [24 ~ 25]
        2 B for create  order (<1024);
        [26 ~ 27] 
        2 B for detailed Size (<1024);
        [28]
        1 B with [23] for modify order, this is 低位;
      1024 KB = store file
    */
  
  // Extra variables within 128 B Extra space
  fs->fileFcbPointer = 0; // [0, 1024]
  fs->freeBlockPointer = 0; // [0, 32K]
  fs->fileOrder = 0;
  fs->modifyOrder = 1;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

}



__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  /* Implement open operation here */
  int fileThisP; // Index of FCB entry of this file [0, 1024]
  bool createNewF = 1; // TRUE if open as write mode and no file found
  int fileThisStartBlock; //记录已打开文件的，从匹配的FCB中获得的 Start Block numebr
  u32 fpReturn; // 存储 return 的fp 值

  fileThisP = fs->fileFcbPointer; // when create new file, FcbPointer ++ 

  // 文件名解析：FCB检索
  outer:
  for(int i=0; i<sizeof(s); i++){ 
    //文件名检索，确认是否已打开文件
    if(i==0 && op==1){ //只有读模式下才会创建空白新文件
      //顺序查找所有FCB条目（因为FCB可能因删除文件而导致中间有空行）
      for(int j=0; j<1024; j++){
        if(fs->volume[4096 + j*32] == s[0]){
          if(s[1] == fs->volume[4096 + j*32 + 1] ){
            // 已经存在同名文件，无需创建
            createNewF = 0;

            // 提取该文件start block信息
            fileThisStartBlock = fs->volume[4096 + j*32 + 20]*255 + fs->volume[4096 + j*32 + 21];
            break outer;
          }
        }
      }
    }

    //创建新文件：文件名存储入FCB
    fs->volume[4096 + fileThisP*32 + i] = s[i]; // FCB fileName store
    fs->fileFcbPointer++;   
  }

  // 创建新的空文件：其它属性存入FCB 对应的 Entry
  if(createNewF){
    // start block
    fs->volume[4096 + fileThisP*32 + 20] = fs->freeBlockPointer / 255;
    fs->volume[4096 + fileThisP*32 + 21] = fs->freeBlockPointer % 255;
    fs->freeBlockPointer++;

    // block length
    fs->volume[4096 + fileThisP*32 + 22] = 1;

    // create order
    fs->volume[4096 + fileThisP*32 + 24] = fs->fileOrder / 255;
    fs->volume[4096 + fileThisP*32 + 25] = fs->fileOrder % 255;
    fs->fileOrder++;
    
    // detailed size
    fs->volume[4096 + fileThisP*32 + 26] = 0;
    fs->volume[4096 + fileThisP*32 + 27] = 0;

    //return (FILE_BASE_ADDRESS + (fs->freeBlockPointer -1)*32); // 1 block = 32 KB

    // 第一位，若FCB index>1000，则最高位加2， 最后三位是fcb的后三位，表示0-999的FCB index，中间是正常的 volumn disk指针
    fpReturn = 2000000000*(fileThisP/1000) + (fs->FILE_BASE_ADDRESS + (fs->freeBlockPointer -1)*32)*1000 + fileThisP%1000;
    return fpReturn;

    // 返回的指针包含 Base Address，可以直接放入 volume[] 数组当中

  }else{
  // 已经存在文件，返回读/写指针

    // return fp: volum[n] array中的准确标号位置n, n从0开始
    fpReturn = 2000000000*(fileThisP/1000) + (fs->FILE_BASE_ADDRESS + fileThisStartBlock*32)*1000 + fileThisP%1000;
    return fpReturn;

    //return (FILE_BASE_ADDRESS + fileThisStartBlock*32); 
  }

  // return 格式:
    // return的u32, 高位用来返回磁盘指针，低位用来返回文件FCB位置便于write&read查询问价对应FCB属性
    // u32 最多表示 42 0000 0000 十进制， FCB entry最多1024，十进制来说留出4位，剩余的 42 0000 则不够 108 5440
    // FCB 留出3位，则 volum 可以又420 0000，而最大只可能是108 000，可以第一位设定特定值如 = 2 & 3 则代表FCB超过1000
    // eg. FCB 1022 = 2/3……022， 检测到第一位>1，则说明FCB的值是大于1000，再减去2 就是原本最高位的值（0 或者 1）
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
  /* Implement read operation here */
  int volumeP;
  int FcbIndex;
  
  //解析fp：read只需要 disk volume的 pointer即可，被读文件属性也要！需要修改modify time！
  if(fp/1000000000 > 1){
    volumeP = (fp-20000000000)/1000;
    FcbIndex = 1000 + fp%1000;
  }else{
    volumeP = fp/1000;
    FcbIndex = fp%1000;
  }

  //读取已给的size大小的内容到output
  for(int i=0; i<size; i++){
    fs->volume[volumeP + i] = output[i]; 
  }

  // 修改被读取文件的modify time
  fs->volume[4096 + FcbIndex*32 + 23] = fs->modifyOrder / 255;
  fs->volume[4096 + FcbIndex*32 + 28] = fs->modifyOrder % 255;
  fs->modifyOrder++;

}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
  /* Implement write operation here */
  
  int FcbIndex; // FCB entry index of opened file [0, 1024]，能直接的通过index找到对应文件的FCB extry
  int volumeP; // original fp data, volume[volumeP] array index, can directly use to locate at array, 一定是某一block的开头

  // 解析 fp
  if(fp/1000000000 > 1){
    FcbIndex = 1000 + fp%1000;
    volumeP = (fp-20000000000)/1000;
  }else{
    FcbIndex = fp%1000;
    volumeP = fp/1000;
  }

  // 首先清除文件中所有的老信息（如果 size 不等于0 说明是一个老文件, 根据length来清除所有 block）
    // FCB中 [26]和[27]表示size， [22]表示 block length
  if((fs->volume[4096 + FcbIndex*32 + 26] + fs->volume[4096 + FcbIndex*32 + 27]) != 0){
    for(int i =0; i< 32*(4096 + fs->volume[4096 + FcbIndex*32 + 22]); i++){
      fs->volume[volumeP + i] = 0;
    }
  } 

  // 写入input buffer中的内容
  for(int j=0; j<size; j++){
    fs->volume[volumeP + j] = input[j];
  }

  // 修改fcb中的数据
  // modify times:
  fs->volume[4096 + FcbIndex*32 + 23]++;
  // size
  fs->volume[4096 + FcbIndex*32 + 26] = size / 255;
  fs->volume[4096 + FcbIndex*32 + 27] = size % 255;
  // block length
  fs->volume[4096 + FcbIndex*32 + 22] = size / 32 + 1;
  // modify time
  fs->volume[4096 + FcbIndex*32 + 23] = fs->modifyOrder / 255;
  fs->volume[4096 + FcbIndex*32 + 28] = fs->modifyOrder % 255;
  fs->modifyOrder++;

  // 修改 disk volume pointer （FCB pointer不需要修改，只有在create和delete文件的时候才需要修改）
    //默认当前文件的最后一块就是磁盘已用的最后一块（也就是free block 的前一块），默认合并完成
  fs->freeBlockPointer = (volumeP - fs->FILE_BASE_ADDRESS)/32 + (size / 32 + 1) + 1;
  // 通过fp算出来的start block 应该和存在 fcb中的相等才对
  //if((volumeP - fs->FILE_BASE_ADDRESS)/32 != [22]*255+[23]){printf error}

  // 修改 bit-vector
  for(int i=0; i<(size/32+1); i++){
    fs->volume[((volumeP - fs->FILE_BASE_ADDRESS)/32 + i)/32] |= (1 << ((volumeP - fs->FILE_BASE_ADDRESS)/32 + i)%32); 
  }

}
__device__ void fs_gsys(FileSystem *fs, int op)
{
  /* Implement LS_D and LS_S operation here */
  int propertyOrder[16]; //记录 modifyTime 或 Size
  int createTOrder[16];
  int fcbIndex[16]; //暂定最多输出16个文件，理论上搞一个1024也行，但怕太大了
  int bufferP = 0;
  //uchar fileName[20];

  /* LS_D */
  if(op == 0){
    printf("=== SORT by Modified Time ===\n");

    //检索FCB，获得index以及modify time
    for(int i=0; i<1024; i++){
      if(fs->volume[4096 + i*32] != 0){ // 通过文件名第一位不等于0来判断是否存在文件（【疑问】是否不保险？暂时先这样吧）
        fcbIndex[bufferP] = i;
        propertyOrder[bufferP] = fs->volume[4096 + i*32 + 26]*255 + fs->volume[4096 + i*32 + 27];
        bufferP++;

        if(bufferP > 15){
          printf("List file only show first 16!");
          break;
        }
      }
    }

    //准备printf文件名
    for(int i=0; i<bufferP; i++){ //bufferP 正好记录了文件数量
      int max= 0;
      int maxOne =0;

      for(int j=0; j<bufferP; j++){
        // 遍历：找到modifyTime最大的那个
        if(propertyOrder[j] > max){
          max = propertyOrder[j];
          maxOne = j;
        }
      }

      // 根据index 提取file name 并且printf
      for(int k=0; k<20; k++){
        printf("%d", fs->volume[4096 + fcbIndex[maxOne]*32 + k]);
      }
      printf("\n");

      // 删除已经输出的之前最大值文件
      propertyOrder[maxOne] = 0;
    }
  }
  

  /* LS_S */
  if(op == 1){
    printf("=== SORT by File Size ===\n");

    //检索FCB，获得index以及Size & create order
    for(int i=0; i<1024; i++){
      if(fs->volume[4096 + i*32] != 0){
        fcbIndex[bufferP] = i;
        propertyOrder[bufferP] = fs->volume[4096 + i*32 + 26]*255 + fs->volume[4096 + i*32 + 27];
        createTOrder[bufferP] = fs->volume[4096 + i*32 + 24]*255 + fs->volume[4096 + i*32 + 25];
        bufferP++;

        if(bufferP > 15){
          printf("List file only show first 16!");
          break;
        }
      }
    }

    // 根据size顺序print文件名&Size
    for(int i=0; i<bufferP; i++){ //bufferP 记录FCB中文件数量
      int maxSize = 0;
      int maxOne = 0;

      for(int j=0; j<bufferP; j++){
        // 遍历：找到Size最大, 且 create Time最小 的那个
        if(propertyOrder[j] > maxSize){
          maxSize = propertyOrder[j];
          maxOne = j;
        }else if(propertyOrder[j] == maxSize){
          if(createTOrder[j] < createTOrder[maxOne]){ //相同的这个比之前createTime小，则优先printf
            maxOne = j;
          }
        }
      }

      // 根据index 提取file name，并且printf Name 和 Size
      for(int k=0; k<20; k++){
        printf("%d", fs->volume[4096 + fcbIndex[maxOne]*32 + k]);
      }
      printf("  %d\n", propertyOrder[maxOne]);

      // 删除已经输出的之前最大值文件
      propertyOrder[maxOne] = 0;
    }
  }

}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
  // 删除文件：
    // 1. 修改FCB条目：先查找；整条entry都要删除（FCB pointer暂时先不调整）【疑问】是否要解决FCB条目的hole问题？（暂不解决）
    // 2. 修改volume block：（根据FCB的start和length）合并磁盘块(合并就是覆盖)，修改volume pointer（减去删除的文件block长度）

  int startBlock; //该删除文件的startBlock信息
  int blockLength; //删除文件的length
  int fcbStartB;

  //查找FCB条目，获得文件对应entry
  for(int i=0; i<1024; i++){
    if(fs->volume[4096 + i*32] == s[0]){
      if(s[1] == fs->volume[4096 + i*32 + 1] ){
        //匹配成功FCB entry
        //读取block信息
        startBlock = fs->volume[4096 + i*32 + 20]*255 + fs->volume[4096 + i*32 + 21];
        blockLength = fs->volume[4096 + i*32 + 22];
        //删除该文件的FCB条目
        for(int j=0; j<32; j++){
          fs->volume[4096 + i*32 + j] = 0;
        }
        break;
      }
    }
    // 文件未匹配报错
    if(i == 1023){
      printf("ERROR: No such file when fs_gsys_delete!");
    }
  }

  //覆盖文件block内容:合并磁盘块
    //直接从该文件末尾到 磁盘指针（第一块空闲块）-1 位置，全部往前挪动即可，中间一定是没有空的
    //后续所有文件长度：（freeBlock-start-length）* 32， 移动距离 = blockLength*32
  for(int i=0; i<(fs->freeBlockPointer-startBlock-blockLength)*32; i++){
    fs->volume[fs->FILE_BASE_ADDRESS + (startBlock)*32 + i] = fs->volume[fs->FILE_BASE_ADDRESS + (startBlock+blockLength)*32 + i];
  }

    //后面多出来的部分还要填0: 根据blockLength的长度
  for(int i=0; i<blockLength*32; i++){
    fs->volume[(fs->freeBlockPointer - blockLength)*32 + i] = 0;
  }

  fs->freeBlockPointer - blockLength;

  // 修改bit-vector: 只需要把尾部移动后多的 length长度的 部分填0即可
  for(int i=0; i<blockLength; i++){
    fs->volume[startBlock/32] &= ~(1 << startBlock % 32);
  }

  //修改后续所有文件的FCB：确认一下哪些文件需要修改：start模块大于当前start的才需要修改 
  for(int i=0; i<1024; i++){
    fcbStartB = fs->volume[4096 + i*32 + 20]*255 + fs->volume[4096 + i*32 + 21];
    if(fcbStartB > startBlock){
      fcbStartB = fcbStartB - blockLength;
      //重新修改start值，等于原值减去 删除文件的 block length
      fs->volume[4096 + i*32 + 20] = fcbStartB / 255;
      fs->volume[4096 + i*32 + 21] = fcbStartB % 255;
    }
  }

}
