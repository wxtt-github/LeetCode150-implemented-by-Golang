## LeetCode150题(Go语言实现，最低复杂度方法)

[题单:Leetcode150题](https://leetcode.cn/studyplan/top-interview-150/)


- [数组/字符串](#数组字符串)
- [双指针](#双指针)
- [滑动窗口](#滑动窗口)
- [矩阵](#矩阵)
- [哈希表](#哈希表)
- [区间](#区间)
- [栈](#栈)
- [链表](#链表)
- [二叉树](#二叉树)
- [二叉树层次遍历](#二叉树层次遍历)
- [二叉搜索树](#二叉搜索树)

### 数组/字符串

1. 合并两个有序数组

```go
func merge(nums1 []int, m int, nums2 []int, n int)  {
    /*思路:从后往前,每次比较末尾元素,较大的放在nums1的末尾,
    同时,为了满足题意,当两元素相等时,优先放入nums2的元素
    */
    for i,j,k := m-1,n-1,m+n-1;j >= 0;k--{
        if i >= 0 && nums1[i] > nums2[j]{
            nums1[k] = nums1[i]
            i--
        } else{
            nums1[k] = nums2[j]
            j--
        }
    }
}``
```

2. 移除元素

```go
func removeElement(nums []int, val int) int {
	/*思路:使用首尾双指针,首指针left寻找等于val的值,尾指针right寻找
	  不等于val的值,各找到一个后进行交换,循环方式类似快排
	*/
	left := 0
	right := len(nums) - 1
	for left <= right {
		for left <= right && nums[left] != val {
            left++
		}
        for left <= right && nums[right] == val{
            right--
        }
        if left <= right{
            temp := nums[left]
            nums[left] = nums[right]
            nums[right] = temp
            left++
            right--
        }
	}
    return left
}
```

3. 删除有序数组中的重复项

```go
func removeDuplicates(nums []int) int {
	/*思路:使用快慢双指针,初始时慢指针指向0,快指针指向1,
	  我们认为慢指针及之前的元素都是不重复的,基于这个思想,
	  让快指针寻找到与慢指针不同的元素,然后慢指针加1,赋值即可
	*/
	len := len(nums)
    if len == 1{
        return 1
    }

	slow := 0
	quick := 1
	for quick < len {
		for quick < len && nums[quick] == nums[slow] {
			quick++
		}
		if quick < len {
			slow++
			nums[slow] = nums[quick]
			quick++
		}
	}
	return slow + 1
}
```

4. 删除有序数组中的重复项II

```go
func removeDuplicates(nums []int) int {
    /*思路:同上题,采用快慢双指针思路,慢指针slow及之前的元素为重复但不超过
    2次的元素,而快指针quick用于寻找重复不超过两次的元素,可以将quick与slow,slow-1
    做对比,若相等则跳过,其余同上题
    */
    slow := 1
    quick := 2
    length := len(nums)
    if length == 1{
        return 1
    }
    if length == 2{
        return 2
    }

    for slow <= quick && quick < length{
        for quick < length && nums[quick] == nums[slow] && nums[quick] == nums[slow-1]{
            quick++
        }
        if quick < length{
            slow++
            nums[slow] = nums[quick]
            quick++
        }
    }
    return slow+1
}
```

5. 多数元素

```go
func majorityElement(nums []int) int {
	/*思路:由于数组总是非空的且总存在多数元素,采用摩尔投票法,
	  首先定义票数最多的元素为major为第一个元素,其count为1,
	  遍历之后的数组,若与major元素相同,则count++,若不同,则count--,
	  当count为0时,若不同,则修改major元素,这是基于总存在多数元素,
	  因此可以这么修改得出正确结果
	*/
	major := nums[0]
	cnt := 1
	for i := 1; i < len(nums); i++ {
        if nums[i] == major{
            cnt++
        } else if nums[i] != major && cnt == 0{
            major = nums[i]
            cnt = 1
        } else{
            cnt--
        }
	}
    return major
}
```

### 双指针

1. 验证回文串

`注:将字符串改为大写的函数为strings.ToUpper(<string>)`

```go
func isPalindrome(s string) bool {
	/*思路:使用首尾双指针,用strings.ToLower(s)先将字符串转为小写,
	  然后判断双指针指向的是否有效,若无效做进行自增自减操作即可
	*/
    s = strings.ToLower(s)
	left := 0
	right := len(s) - 1
	for left < right {
        if !isValid(s[left]){
            left++
            continue
            //continue的目的是重新进入for循环,这样能实现一直++直到出现有效字符的功能
        }
        if !isValid(s[right]){
            right--
            continue
        }
        if s[left] != s[right]{
            return false
        }
        left++
        right--
	}
    return true
}

func isValid(c byte) bool {
	if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') {
		return true
	}else{
        return false
    }
}
```

2. 判断子序列

```go
func isSubsequence(s string, t string) bool {
    /*思路1:双指针,分别指向子串s和主串t,若能匹配则两指针均往后移,
    若无法匹配则只有主串的指针往后移
    思路2:貌似还有动态规划求解,我觉得双指针比较简单就不写了
    */
    ps := 0
    pt := 0
    for ps < len(s) && pt < len(t){
        if s[ps] == t[pt]{
            ps++
            pt++
        }else{
            pt++
        }
    }
    if ps == len(s){
        return true
    }else{
        return false
    }
}
```

3. 两数之和 II - 输入有序数组

```go
func twoSum(numbers []int, target int) []int {
    /*思路:采用首尾双指针,因为序列是非递减的且题目说明了必有答案,
    因此可计算首尾指针之和sum,若sum小于target,说明sum不够大,因此将首指针后移,
    若sum过小,则将尾指针前移,其中的原理在于,如首指针所指的值过小,则前面的也均不匹配
    */
    left := 0
    right := len(numbers)-1
    for left < right && numbers[left]+numbers[right] != target{
        sum := numbers[left] + numbers[right]
        if sum < target{
            left++
        }else if sum > target{
            right--
        }
    }
    array := []int{left+1,right+1}
    return array
}
```

4. 盛最多水的容器

```go
func maxArea(height []int) int {
    /*思路:使用首尾双指针,首先对于初始状态,宽是最大的,之后宽只会越变越小,
    这时候我们根据双指针的高度,尽可能抛弃较小的高度,直到双指针会合停止
    */
    left := 0
    right := len(height)-1
    var MaxArea int = 0
    for left < right{
        tempArea := (right-left) * min(height[left],height[right])
        MaxArea = max(MaxArea,tempArea)
        if height[left] < height[right]{
            left++
        }else{
            right--
        }
    }
    return MaxArea
}

func min(a,b int) int{
    if a < b{
        return a
    }else{
        return b
    }
}
func max(a,b int) int{
    if a > b{
        return a
    }else{
        return b
    }
}
```

5. 三数之和

```go
func threeSum(nums []int) [][]int {
    /*思路:首先要把三数之和问题转化为两数之和问题,固定一个数a,
    则另外两个数b和c之和等于-a时,三数之和为0,由此转化为两数之和问题。
    首先对数组进行排序，分情况讨论，遍历数组每一个元素，若第一个数大于0，
    则说明不可能存在三数之和为0的数，直接返回。由于题目要求不重复的三元组，
    因此除了第一个数以外，可以检查当前数是否与上一个数相等，若相等则continue，
    若不相等则用双指针解决两数之和问题，注意两数之和同样要去重
    */
    sort.Ints(nums)
    var array [][]int
    if len(nums) < 3{
        return array
    }
    for index,value := range nums{
        if value > 0{
            return array
        }
        if index > 0 && nums[index] == nums[index-1]{
            continue
        }
        left := index+1
        right := len(nums)-1
        for left < right{
            sum := value + nums[left] + nums[right]
            if sum == 0{
                array = append(array,[]int{value,nums[left],nums[right]})
                for left < right && nums[left] == nums[left+1]{
                    left++
                }
                for left < right && nums[right] == nums[right-1]{
                    right--
                }
                left++
                right--
            }else if sum < 0{
                left++
            }else if sum > 0{
                right--
            }
        }
    }
    return array
}
```

### 滑动窗口

1. 长度最小的子数组

```go
func minSubArrayLen(target int, nums []int) int {
    /*思路：采用滑动窗口的思想，首先定义两个指针start和end，均指向0，
    然后定义sum，先让sum加上end指的内容，若此时sum满足大于等于target，
    则与min长度进行比较，若小于则进行更新，不小于则不管，若sum依然小于target，
    则让end后移，同时加上sum加上end的内容，若sum大于等于target，
    则还要将start往后移，并减去start所指的内容，由此得到结果
    */
    start := 0
    end := 0
    length := len(nums)
    minLength := length+1
    if len(nums) == 0{
        return 0
    }
    sum := 0
    for start < length && end < length{
        sum += nums[end]
        for sum >= target{
            minLength = min(minLength,end-start+1)
            sum-=nums[start]
            start++
        }
        end++
    }
    if minLength == length+1{
        return 0
    }else{
        return minLength
    }
}

func min(x int,y int) int{
    if x < y{
        return x
    }else{
        return y
    }
}
```

2. 无重复字符的最长子串

```go
func lengthOfLongestSubstring(s string) int {
	/*思路：使用滑动窗口的思想，首先定义两个指针i和right，
	  i初始指向0，right初始指向-1，定义一个哈希集合，
	  m := map[byte]int{}，用于记录每个字符出现次数，
	  将i从0遍历到s的长度，若i不为0，首先进行delete(m,s[i-1])操作，
	  （因为在for循环里i++），然后不断右移right指针，直到找到不重复出现的字符，
	  此时不重复子串的长度为right-i+1，再将其与原先的max长度相比较即可
      
      时间复杂度：O(n)
      空间复杂度：O(ASCLL码的范围大小)
	*/
	m := map[byte]int{}
	length := len(s)
	maxLength := 0
	right := -1
	for i := 0; i < length; i++ {
        if i != 0{
            delete(m,s[i-1])
        }
        for right+1 < length && m[s[right+1]]==0{
            m[s[right+1]]++
            right++
        }
        maxLength = max(maxLength,right-i+1)
	}
    return maxLength
}

func max(x,y int) int {
    if x < y{
        return y
    }else{
        return x
    }
}
```

3. 串联所有单词的子串(2ms，得意的一题)

```go
func findSubstring(s string, words []string) []int {
	/*思路：这是一道难度很高的题，采用滑动窗口的思想，首先定义一个map，
	  存放words中各个单词出现的次数。由于words中各个元素长度k相等，因此
	  我们将s进行划分处理，每次对单个元素的长度进行增加，一个个单词地处理。
	  首先定义一个for循环，i取0，i小于单个元素长度k，i自增，因为后续还有一层
	  for循环，以k自增，因此为了避免重复。i从0到k-1循环即可。
	  定义计数变量cnt初值为0，定义一个map，名叫counterMap，
	  定义一个左指针left用于记录初始位置方便后续append进数组里，
	  定义一个右指针right用于进行取单词处理，两者初值均为i，
	  right满足right<=len(s)-k，每次自增k，取单词word:=s[right:right+k]，
	  用一个if num,found := m[word];found{}，num代表word的值，found代表是否
	  查找成功，因此当查找成功时，我们首先判断counterMap中word的值是否大于等于num，
	  若大于等于，则说明单词数超标了，将counterMap中s[l:l+k]减1，cnt--，l+=k，
	  这步是为了防止比如words有"a"和"b"，这时候需要的是一个a和一个b，不是两个a。
	  然后将counterMap[word]++，count++。
	  当查找失败时，说明当前单词不在map里，将left指针右移至right+k的位置，同时清除
	  counterMap中的元素，cnt--，最后由于是for循环，right会+=k,因此又会从left==right
	  的情况开始。
	  做完上面两个判断后，判断cnt是否等于words的长度，若等于则将left指针append进数组中
	*/
	arr := []int{}
	lenOfS := len(s)
	lenOfWords := len(words)
	if lenOfWords == 0 {
		return arr
	}
	lenOfSingle := len(words[0])
	if lenOfS < lenOfWords*lenOfSingle {
		return arr
	}

	m := map[string]int{}
	for _, temp := range words {
		m[temp]++
	}
	for i := 0; i < lenOfSingle; i++ {
        cnt := 0
        counterMap := map[string]int{}
        for left,right := i,i;right <= lenOfS-lenOfSingle;right+=lenOfSingle{
            tempWord := s[right:right+lenOfSingle]
            if num,found := m[tempWord];found{
                for counterMap[tempWord] >= num{
                    counterMap[s[left:left+lenOfSingle]]--
                    cnt--
                    left+=lenOfSingle
                }
                counterMap[tempWord]++
                cnt++
            }else{
                for left < right{
                    counterMap[s[left:left+lenOfSingle]]--
                    cnt--
                    left+=lenOfSingle
                }
                left+=lenOfSingle
            }
            if cnt == lenOfWords{
                arr = append(arr,left)
            }
        }
	}
    return arr
}
```

4. 最小覆盖子串

```go
func minWindow(s string, t string) string {
	/*思路：滑动窗口思想，定义两指针left和right，包裹起来的为滑动窗口，
	  主体思路是先使right右移，直到找到一个可行解，然后再让left右移，
	  若右移left依然能够包括t，则继续右移，直到无法包括，此时找到了一个局部最优解，
	  找到局部最优解后，用该局部最优解去尝试更新最优解，然后将左指针右移一位，使得该
	  窗口刚好不能包含t中所有字符，开始下一轮寻找。
	  具体实现是首先定义一个map，map中byte的值负数代表相比较t缺了多少个，0表示正好，
	  正数表示相比较t多了多少个。首先对t遍历，将map中的值进行自减，定义diff作为map中
	  byte的种类。
	  然后定义一个for循环，left和right初值为0，right<s的长度，每次right++，
	  设置if对s[right]在map中查找，查找成功则将其值++，若正好++后为0，则diff--。
	  其中再写一个for，条件是diff==0，每次left++，设置if对s[left]在map中查找，
	  若查找成功则将其值--，若正好减为-1，则将diff++，此时找到局部最优解，
	  若长度小于最短长度，则将字符串改为s[left:right+1]，长度更新为right-left+1。
      时间复杂度：O(n)，n为s长度
      空间复杂度：O(m)，m为t中字符种类
	*/
	str := ""
	minLength := len(s) + 1
	m := make(map[byte]int)
	lenOfT := len(t)
	for i := 0; i < lenOfT; i++ {
		m[t[i]]--
	}
	diff := len(m)
	for left, right := 0, 0; right < len(s); right++ {
        if _,found := m[s[right]];found{
            m[s[right]]++
            if m[s[right]] == 0{
                diff--
            }
        }
        for diff == 0{
            if _,found := m[s[left]];found{
                m[s[left]]--
                if m[s[left]] == -1{
                    diff++
                    if right-left+1 < minLength{
                        minLength = right-left+1
                        str = s[left:right+1]
                    }
                }
            }
            left++
        }
	}
    return str
}
```

### 矩阵

1. 有效的数独

```go
func isValidSudoku(board [][]byte) bool {
	/*思路：对于一个i行j列的数组元素，如果他不是'.'，采用计数的思想，
	  创建两个9*9的二维矩阵row和column，第一个9用于表示多少行或列，
	  第二个9用于表示数字几出现的次数，如row[0][0]=2代表第1行的1元素
	  出现了两次，因此不符题意，返回true。
	  本题难点在每一个3*3的小九宫格sub，创建一个3*3*9的三维矩阵，
	  前两个3表示第几行第几列的小九宫格（把每个小九宫格看做整体），
	  后面的9代表数字几出现的次数，如sub[0][0][1]=0代表第一行第一列的
	  小九宫格中1出现了0次。
	  因此对于i行j列的元素m，我们只需要将row[i][m-1]++，column[j][m-1]++，
	  sub[i/3][j/3][m-1]++即可，若出现其中某个大于1时，直接返回false
	*/
	rows := [9][9]int{}
    //还有一种等价写法是var rows [9][9]int，其他的make，或者省略{}都是不合法的
	columns := [9][9]int{}
	subs := [3][3][9]int{}
	for i, row := range board {
		for j, val := range row {
			if val == '.' {
				continue
			}
			index := val - '1'
			rows[i][index]++
			columns[j][index]++
			subs[i/3][j/3][index]++
			if rows[i][index] > 1 || columns[j][index] > 1 || subs[i/3][j/3][index] > 1 {
                return false
			}
		}
	}
    return true
}
```

2. 螺旋矩阵

```go
func spiralOrder(matrix [][]int) []int {
	/*思路：模拟螺旋过程，定义left，right，top，bottom，初值分别为
		0，len(matrix[0])-1,0,len(matrix)-1，其中len(matrix)代表matrix有几行，
		然后遵循从左往右，从上往下，从右往左，从下往上的顺序，最外层for循环终止条件
		是遍历完(right+1)*(bottom+1)个元素，用number来控制遍历完这些元素，防止继续
	    进行for循环
        时间复杂度：O(矩阵元素数)
        空间复杂度：不算arr的话O(1)，算的话O(矩阵元素数)
	*/
	arr := []int{}
	left, right, top, bottom := 0, len(matrix[0])-1, 0, len(matrix)-1
	number := (right + 1) * (bottom + 1)
	i := 0
	for i < number {
		for j := left; j <= right && i < number; j++ {
			arr = append(arr, matrix[top][j])
			i++
		}
		top++
		//从左往右，贴着上墙走
		for j := top; j <= bottom && i < number; j++ {
			arr = append(arr, matrix[j][right])
			i++
		}
		right--
		//从上往下，贴着右墙走
		for j := right; j >= left && i < number; j-- {
			arr = append(arr, matrix[bottom][j])
			i++
		}
		bottom--
		//从右往左，贴着下墙走
		for j := bottom; j >= top && i < number; j-- {
			arr = append(arr, matrix[j][left])
			i++
		}
		left++
		//从下往上，贴着左墙走
	}
	return arr
}
```

3. 旋转图像

```go
func rotate(matrix [][]int)  {
    /*思路：顺时针旋转90度相当于先垂直翻转（以水平线为轴上下翻转），
    再对角线翻转。同理旋转180,270只需要多旋转几次即可
    */
    n := len(matrix)
    for i := 0;i < n/2;i++{
        for j := 0;j < n;j++{
            matrix[i][j],matrix[n-1-i][j] = matrix[n-1-i][j],matrix[i][j]
        } 
    }
    //垂直翻转
    for i := 0;i < n;i++{
        for j := i;j < n;j++{
            matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
        }
    }
    //对角线翻转
}
```

4. 矩阵置零

```go
func setZeroes(matrix [][]int) {
	/*思路：对于一个矩阵，我们先不看第0行和第0列，
	  从第1行第1列以后的子矩阵往后看，对其遍历元素i，j，若找到0，则可以标记
	  其i行0列，0行j列的元素为0，之后我们重新遍历元素i，j，若发现其i行0列或
	  0行j列的元素为0，则将其置为0。这样我们可以利用第0行和0列的空间来储存信息。
	  但是这样我们处理了1行1列后的子矩阵，第0行和第0列没解决，因此我们一开始就
	  遍历第0行和第0列，设置两个变量用来记录是否存在0，再根据这个变量来处理0行0列
	  时间复杂度：O(mn)
	  空间复杂度：O(1)
	*/
	rows := len(matrix)
	columns := len(matrix[0])
	isRowExist0 := false
	isColumnExist0 := false

	for i := 0; i < columns; i++ {
		if matrix[0][i] == 0 {
			isRowExist0 = true
			break
		}
	}
	for i := 0; i < rows; i++ {
		if matrix[i][0] == 0 {
			isColumnExist0 = true
			break
		}
	}
	//判断第0行0列是否有0
	for i := 1; i < rows; i++ {
		for j := 1; j < columns; j++ {
			if matrix[i][j] == 0 {
				matrix[0][j] = 0
				matrix[i][0] = 0
			}
		}
	}
	for i := 1; i < rows; i++ {
		for j := 1; j < columns; j++ {
			if matrix[i][0] == 0 || matrix[0][j] == 0 {
				matrix[i][j] = 0
			}
		}
	}
	//处理1行1列后的子矩阵
	if isRowExist0 {
		for i := 0; i < columns; i++ {
			matrix[0][i] = 0
		}
	}
	if isColumnExist0 {
		for i := 0; i < rows; i++ {
			matrix[i][0] = 0
		}
	}
}
```

5. 生命游戏

```go
func gameOfLife(board [][]int)  {
    /*思路：首先得写一个countCell函数，用于计算周围8个方向的活细胞数，
    为了实现这个函数，可以定义两个方向数组dx和dy，以向下和向右为正，
    定义8个方向,需要注意最后计算数目时要&1，目的是取到最后一位数字（当前状态）。
    我们以1代表细胞活，0代表细胞死，为了原地实现算法，我们不能即时更新
    01的变化，而要借助其他数字2和3。我们以[下一状态，当前状态]的格式定义。
    00：死到死
    01：生到死
    10：死到生
    11：生到生
    这样定义有一个好处，我们先假意更新成2和3，最后再整体更新一次，借助
    右移运算，我们发现能正好保留高位数字，也就是更新成下一状态。
    因此总体思路是先遍历一遍，更新成0,1,2,3的数字（其实只需要关注2,3，
    因为原先就是01,01不变也不影响，目的是把01变为23），
    然后再遍历一遍，进行右移运算即可。
    */
    if len(board) == 0{
        return
    }
    rows := len(board)
    columns := len(board[0])
    for i := 0;i < rows;i++{
        for j := 0;j < columns;j++{
            cnt := countCell(board,i,j)
            if board[i][j] == 0 && cnt == 3{
                board[i][j] = 2
            }else if board[i][j] == 1 && (cnt == 2 || cnt == 3){
                board[i][j] = 3
            }
        }
    }
    for i := 0;i < rows;i++{
        for j:= 0;j < columns;j++{
            board[i][j]>>=1
        }
    }
}
func countCell(board [][]int,x,y int) int{
    dx := [8]int{0,0,-1,1,1,1,-1,-1}
    dy := [8]int{-1,1,0,0,-1,1,-1,1}
    //上，下，左，右，东北，东南，西北，西南
    rows := len(board)
    columns := len(board[0])
    cellNumber := 0
    for i := 0;i < 8;i++{
        cx := x+dx[i]
        cy := y+dy[i]
        if cx<0 || cx>=rows || cy<0 || cy >=columns{
            continue
        }else{
            cellNumber+=board[cx][cy]&1
        }
    }
    return cellNumber
}
```

### 哈希表

1. 赎金信

```go
func canConstruct(ransomNote string, magazine string) bool {
    /*思路：定义一个哈希表m，遍历magazine，对其键值对自增，
    然后遍历ransomNote，将其键值对自减，若减为0，则删除
    时间复杂度：O(m+n)
    空间复杂度：O(1)，因为哈希表最多存26个字母（根据题目要求）
    */
    m := map[byte]int{}
    for i := 0;i < len(magazine);i++{
        m[magazine[i]]++
    }
    for i := 0;i < len(ransomNote);i++{
        _,found := m[ransomNote[i]]
        if found{
            m[ransomNote[i]]--
            if m[ransomNote[i]] == 0{
                delete(m,ransomNote[i])
            }
        }else{
            return false
        }
    }
    return true
}
```

2. 同构字符串

```go
func isIsomorphic(s string, t string) bool {
    /*思路：创建一个哈希表m1，类型map[byte]byte，同时遍历s和t，
    s为key，t为值，每次现在哈希表中寻找，若寻找失败则新建键值对，
    若寻找成功则用t去匹配map[s]，若不匹配则返回false。
    但是这样会遇到一个问题，如badc映射到baba是错误的，1个哈希表
    只能维护单向映射，而无法完成一对一关系，因此需要两个哈希表。
    时间复杂度：O(n)
    空间复杂度：O(1)，因为7位ASCII码最多也就128个字符
    */
    m1 := map[byte]byte{}
    m2 := map[byte]byte{}
    for i := 0;i < len(s);i++{
        val1,found1 := m1[s[i]]
        val2,found2 := m2[t[i]]
        if found1{
            if val1 != t[i]{
                return false
            }
        }else{
            m1[s[i]] = t[i]
        }

        if found2{
            if val2 != s[i]{
                return false
            }
        }else{
            m2[t[i]] = s[i]
        }
    }
    return true
}
```

3. 单词规律

```go
func wordPattern(pattern string, s string) bool {
    /*思路：和上一题字符映射差不多，建立两个哈希表维护一对一关系即可，
    难点是将s进行字符串分割，通过words := strings.Split(s," ")，
    即可得到一个以空格进行分割的字符串切片。注意在判断m2时，判断不为空
    是用>0来判断，而不是!=''，这是因为byte的默认值为0，可以自己输出一下验证。
    此外提供了map的三种声明方式，需注意用var
    声明时必须用make创建。
    时间复杂度：O(n)
    空间复杂度：O(1)。因为对于pattern来说，最多就128种字符
    */
    words := strings.Split(s," ")
    if len(words) != len(pattern){
        return false
    }
    var m1 map[byte]string = make(map[byte]string)
    var m2 map[string]byte = make(map[string]byte)
    // m1 := map[byte]string{}
    // m2 := map[string]byte{}
    // m1 := make(map[byte]string)
    // m2 := make(map[string]byte)
    for i := 0;i < len(pattern);i++{
        ch := pattern[i]
        word := words[i]
        if (m1[ch] != "" && m1[ch] != word) || (m2[word] > 0 && m2[word] != ch){
            return false
        }
        m1[ch] = word
        m2[word] = ch
    }
    return true
}
```

4. 有效的字母异位词

```go
func isAnagram(s string, t string) bool {
    /*思路：建立一个哈希表m，对s遍历，然后再对t遍历，
    若查找失败直接返回false，若查找成功则进行自减
    时间复杂度：O(n)
    空间复杂度：O(1)
    */
    m := make(map[byte]int)
    if len(s) != len(t){
        return false
    }
    for i := 0;i < len(s);i++{
        m[s[i]]++
    }
    for i := 0;i < len(t);i++{
        _,found := m[t[i]]
        if found{
            m[t[i]]--
            if m[t[i]] == 0{
                delete(m,t[i])
            }
        }else{
            return false
        }
    }
    return true
}
```

5. 字母异位词分组

```go
func groupAnagrams(strs []string) [][]string {
    /*思路：判断两词是否是字母异位词很简单，本题难点是如何
    同时对多个词处理，并把它们组合到一起。可以定义一个哈希表，
    m := map[[26]int][]string{}。其中key是[26]int，
    通过这个数组我们可以唯一确定一种字母异位词，val是[]string类型，
    一个字符串数组。我们遍历strs中的每一个字符串，计算它们各自的[26]int，
    然后把相同key的字符串加入到字符串数组中，即可完成。
    注意：用[26]int来唯一标记时，访问下标要进行-'a'操作，变为0-25的数字
    时间复杂度：O(n(k+Σ))，n是strs中字符串的数量，k是单个字符串最大长度，Σ是字符集长度
    空间复杂度：O(n(k+Σ))
    */
    m := map[[26]int][]string{}
    arr := [][]string{}
    for _,str := range strs{
        cnt := [26]int{}
        for _,ch := range str{
            cnt[ch-'a']++
        }
        m[cnt] = append(m[cnt],str)
    }
    for _,val := range m{
        arr = append(arr,val)
    }
    return arr
}
```

6. 两数之和

```go
func twoSum(nums []int, target int) []int {
    /*思路：有两种方法，一种是暴力枚举法，一种是哈希表法。
    法一：两个for循环暴力枚举
    时间复杂度：O(n^2)
    空间复杂度：O(1)
    法二：建立一个哈希表m，类型map[int]int，key为值，val为下标，
    遍历nums中每一个元素时，首先先在哈希表中寻找map[target-num]，
    若找到则直接返回，若找不到则将其加入哈希表中
    时间复杂度：O(n)
    空间复杂度：O(n)
    更推荐的是第二种方法，注释掉的是第一种方法
    */
    m := map[int]int{}
    for i,val := range nums{
        if j,found := m[target-val];found{
            return []int{i,j}
        }else{
            m[val] = i
        }
    }
    return []int{}


    // for i := 0;i < len(nums)-1;i++{
    //     for j := i+1;j < len(nums);j++{
    //         if nums[i]+nums[j] == target{
    //             return []int{i,j}
    //         }
    //     }
    // }
    // return []int{}
    // //法一
}
```

7. 快乐数

```go
func isHappy(n int) bool {
    /*思路：首先要写一个计算各位平方和的函数，这个不难，
    定义sum=0，当n>0时，每次取最后一位，计算平方加进sum中，
    重复循环即可。
    对于一个具体的数，它可能是快乐数，也可能不是快乐数。
    若是快乐数，那它总有一天会变为1，如先变为100，再变为1。
    若不是快乐数，那它最终一定会陷入一个循环，为什么不是趋向于
    无穷大呢？如对于999会变为243，对于9999会变为324，这说明
    对于一个很大的高位数来说，他会先降维到3位数，3位数又会降维
    成243及以下的数，而这是一个有限的集合，说明他最终一定会陷入循环。
    法一：哈希表法，可以定义一个哈希表，记录n的变化路径，若n变为1或者n变为
    哈希表曾经出现过的数，说明其陷入了循环，跳出，判断n==1即可。
    时间复杂度：O(logn)
    空间复杂度：O(logn)
    法二：快慢双指针法，定义快慢双指针fast和slow，如果一个数是快乐数，
    那么fast肯定最先变为1，若不是快乐数，那这两个指针会在某一个循环的
    数相遇
    时间复杂度：O(logn)
    空间复杂度：O(1)
    从极致的角度看，更推荐法二，但法一也不错了，注释掉的是法一
    */
    slow,fast := n,cal(n)
    for fast != 1 && slow != fast{
        fast = cal(cal(fast))
        slow = cal(slow)
    }
    return fast == 1
    // 法二


    // m := map[int]bool{}
    // for n != 1 && !m[n]{
    //     m[n] = true
    //     n = cal(n)
    // }
    // return n == 1
    // //法一
}
func cal(n int)int{
    sum := 0
    for n > 0{
        sum += (n%10) * (n%10)
        n/=10
    }
    return sum
}
```

8. 存在重复元素II

```go
func containsNearbyDuplicate(nums []int, k int) bool {
    /*思路：首先先实现一个求绝对值距离的函数(或者顺便判断小于等于k)，
    然后创建一个哈希表，遍历nums，若查找成功，则计算其距离并进行更新
    时间复杂度：O(n)
    空间复杂度：O(n)
    */
    m := map[int]int{}
    if len(nums) < 2{
        return false
    }
    for i,num := range nums{
        val,found := m[num]
        if found && absDistance(val,i) <= k{
            return true
        }else{
            m[num] = i
        }
    }
    return false
}
func absDistance(x,y int)int{
    temp := x - y
    if temp > 0{
        return temp
    }else{
        return -temp
    }
}
```

9. 最长连续序列

```go
func longestConsecutive(nums []int) int {
    /*思路：题目要求O(n)时间复杂度，意味着只能遍历常数次nums，
    且不能用sort.Ints(nums)来排序。
    首先定义一个哈希表m，遍历一遍nums，目的是去重。然后，重新
    遍历nums，每遍历一个num元素，首先查找m[num-1]是否存在，
    若存在说明它不是连续序列的开始元素，我们将其跳过。
    若不存在说明它是一个连续序列的开始元素（尽管这个序列长度可能为1）。
    在得知它是开始元素后，就好办了，依次查找m[num+1]，m[num+2]，
    直到查找失败，得出这个连续序列的长度，再与已有的最大长度进行比较即可。
    时间复杂度：O(n)。虽然有两个for，但是可以假想有一个长度为n的连续序列，
    当找到开始元素时，第二个for只需遍历剩下n-1个元素，而对于第一个for，
    这n-1个元素直接会跳过第二个for，因此是O(n)
    空间复杂度：O(n)
    */
    if len(nums) == 0{
        return 0
    }
    m := make(map[int]bool)
    maxLength := 0
    for _,val := range nums{
        m[val] = true
    }
    for key := range m{
        if found := m[key-1];!found{
            currentLength := 1
            temp := key+1
            ok := m[temp]
            for ok{
                currentLength++
                temp++
                ok = m[temp]
            }
            maxLength = max(maxLength,currentLength)
        }
    }
    return maxLength
}
func max(x,y int)int{
    if x < y{
        return y
    }else{
        return x
    }
}
```

### 区间

1. 汇总区间

```go
func summaryRanges(nums []int) []string {
    /*思路：
    法一：建立一个哈希表，先遍历一遍nums，然后寻找
    每个序列的开始元素，若找到开始元素，遍历这个序列，即可。
    将int转为string可用strconv.Itoa(x int) string{}函数
    时间复杂度：O(n)
    空间复杂度：O(n)
    法二：由于题目给的是一个无重复元素的有序数组，法一并没有
    用到这个性质，可以定义双指针left和right，left记录区间起点，
    而right记录区间终点，用right不断右移直到找到终点，添加进字符串
    数组即可
    时间复杂度：O(n)
    空间复杂度：O(1)。如果不算输出的数组的话，而法一有哈希表必是O(n)
    推荐法二，注释掉的是法一
    */
    arr := []string{}
    length := len(nums)
    left,right := 0,0
    for right < length{
        left = right
        right++
        for right < length && nums[right-1]+1 == nums[right]{
            right++
        }
        if right-1 == left{
            arr = append(arr,strconv.Itoa(nums[left]))
        }else{
            str := strconv.Itoa(nums[left]) + "->" + strconv.Itoa(nums[right-1])
            arr = append(arr,str)
        }
    }
    return arr
    //法二


    // m := map[int]bool{}
    // for _,val := range nums{
    //     m[val] = true
    // }
    // arr := []string{}
    // for _,val := range nums{
    //     if found := m[val-1];!found{
    //         temp := val+1
    //         ok := m[temp]
    //         for ok{
    //             temp++
    //             ok = m[temp]
    //         }
    //         if temp-1 == val{
    //             var str string = strconv.Itoa(val)
    //             arr = append(arr,str)
    //         }else{
    //             var str string = strconv.Itoa(val) + "->" + strconv.Itoa((temp-1))
    //             arr = append(arr,str)
    //         }
    //     }
    // }
    // return arr
    // //法一
}
```

2. 合并区间

```go
func merge(intervals [][]int) [][]int {
    /*思路：这个问题的前置条件是知道如何将多维数组排序，使用这个接口，
    sort.Slice(arr [][]int,func(i, j int) bool{})进行排序，
    注意括号的位置，arr是待排序数组，func是制定的规则，如通过在{}
    里书写return arr[i][0] < arr[j][0]，即可通过起始点进行排序，
    排序方式为从小到大。这个函数采用快排，所以时间复杂度是O(nlogn)，
    空间复杂度是O(logn)(不算返回数组的话)。
    首先我们将该数组按起始点坐标进行排序，先加入该数组0号区间，作为
    已经合并好的区间，然后对剩余区间的第一个进行判断，若该区间的左端点
    大于已合并好的末尾区间的右端点，则说明它们不重合，直接把它加入数组中，
    若小于等于，则说明可以进行合并，将已合并好的末尾区间的右端点改为
    剩余区间的第一个的右端点和末尾区间的右端点的最大值。
    */
    if len(intervals) == 0{
        return intervals
    }
    sort.Slice(intervals,func(i,j int)bool{
        if intervals[i][0] < intervals[j][0]{
            return true
        }else{
            return false
        }
    })
    arr := [][]int{}
    arr = append(arr,intervals[0])
    lastIndex := 0
    for i := 1;i < len(intervals);i++{
        if intervals[i][0] <= arr[lastIndex][1]{
            arr[lastIndex][1] = max(intervals[i][1],arr[lastIndex][1])
        }else{
            arr = append(arr,intervals[i])
            lastIndex++
        }
    }
    return arr
}
func max(x, y int)int{
    if x > y{
        return x
    }else{
        return y
    }
}
```

3. 插入区间

```go
func insert(intervals [][]int, newInterval []int) [][]int {
    /*思路：一种思路是如上题，找到合适的地方插入区间，然后再进行合并，
    但是像上题那样合并的话需要对intervals中的每个区间进行操作，而该题
    给的是一个无重叠的有序区间，可以转换思路，分成三部分处理。
    首先对于intervals中右端点小于新区间左端点的元素，说明其不重叠，
    可以直接append进新的空数组，然后对于与新区间有重叠的元素，不管
    是重叠了一部分，还是完全重叠，都可以进行合并，合并的方法是，直接对
    新区间进行处理，将新区间的左端点改为它们中左端点的最小值，右端点改为
    它们中右端点的最大值，对每一个重叠的元素都进行这个处理，即可完成合并。
    对于intervals中左端点大于新区间右端点的元素，说明其不重叠，直接append
    进新数组中即可。
    时间复杂度：O(n)
    空间复杂度：O(1)，如果算上返回数组的话是O(n)
    */
    arr := [][]int{}
    i := 0
    length := len(intervals)
    for i < length && intervals[i][1] < newInterval[0]{
        arr = append(arr,intervals[i])
        i++
    }
    for i < length && intervals[i][0] <= newInterval[1]{
        newInterval[0] = min(intervals[i][0],newInterval[0])
        newInterval[1] = max(intervals[i][1],newInterval[1])
        i++
    }
    arr = append(arr,newInterval)
    for i < length && intervals[i][0] > newInterval[1]{
        arr = append(arr,intervals[i])
        i++
    }
    return arr
}
func max(x,y int) int{
    if x > y{
        return x
    }else{
        return y
    }
}
func min(x,y int) int{
    if x < y{
        return x
    }else{
        return y
    }
}
```

4. 用最少数量的箭引爆气球

```go
func findMinArrowShots(points [][]int) int {
    /*思路：
    具体参考：https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/?envType=study-plan-v2&envId=top-interview-150
    首先得将数组按右端点升序排列，为什么不按左端点而是按右端点的原因是，
    这样排序后，可以根据下一个区间的左端点和当前区间的右端点进行判断是否
    重合，若重合则把他们归为一起不消耗箭，若不重合则需要多消耗一支箭。
    首先将cnt设为1，然后对points进行遍历，若左端点大于当前的右端点，则
    修改当前的右端点，并将箭的数量加一
    时间复杂度：O(nlogn)
    空间复杂度：O(logn)
    */
    if len(points) == 0{
        return 0
    }
    sort.Slice(points,func(i,j int)bool{
        if points[i][1] < points[j][1]{
            return true
        }else{
            return false
        }
    })
    cnt := 1
    maxRight := points[0][1]
    for _,val := range points{
        if val[0] > maxRight{
            maxRight = val[1]
            cnt++
        }
    }
    return cnt
}
```

### 栈

1. 有效的括号

```go
func isValid(s string) bool {
    /*思路：用栈来解决括号匹配，go里没有原生的栈，需要自己定义。
    定义成一个数组即可，然后遍历s，遇到左括号压栈，遇到右括号
    进行出栈匹配，最后查看栈中是否有残余元素即可。
    注：在用range遍历string时，val要进行强制类型转换，经我测试，
    val的默认类型是uint8
    时间复杂度：O(n)
    空间复杂度：O(n)
    */
    stack := []byte{}
    for _,val := range s{
        ch := byte(val)
        if ch == '(' || ch == '[' || ch == '{'{
            stack = append(stack,ch)
        }else{
            if len(stack) == 0{
                return false
            }
            top := stack[len(stack)-1]
            if ch == ')' && top == '(' || ch == ']' && top == '[' || ch == '}' && top == '{'{
                stack = stack[:len(stack)-1]
            }else{
                return false
            }
        }
    }
    return len(stack)==0
}
```

2. 简化路径

```go
// import path2 "path"
// //用法一时取消注释
// //因为库名和变量名重名了，做点别名处理
func simplifyPath(path string) string {
    /*思路：
    法一：利用标准库，path.Clean(s string) string{}
    法二：用栈来处理，首先将path用/进行分割，然后对该数组进行遍历，
    如果遇到".."，那么进行弹栈，如果遇到既不是""又不是"."，则进行压
    栈，最后再将该数组用"/"来连接，开头再加上"/"。关键是懂得调用
    strings.Split(s string,"/"){}和strings.Join(arr []string,"/"){}
    这两个函数
    时间复杂度：O(n)
    空间复杂度：O(n)
    */
    stack := []string{}
    arr := strings.Split(path,"/")
    for _,val := range arr{
        if val == ".."{
            if len(stack) > 0{
                stack = stack[:len(stack)-1]
            }
        }else if val != "" && val != "." && val != ".."{
            stack = append(stack,val)
        }
    }
    return "/"+strings.Join(stack,"/")
    //法二


    // return path2.Clean(path)
    // //法一
}
```

3. 最小栈

```go
/*思路：本题一方面要懂得如何构建go的类和方法，二要解决在O(1)时间
        得到栈中最小元素的问题。在定义MinStack时，可以用一个数组
        来初始化栈，用另一个数组模拟栈记录栈的最小元素，我们每次压栈，
        同时也把当前最小值压入最小元素栈，这样检索最小元素时，只需要
        查看最小元素栈的栈顶即可，还要注意在pop时，也要同时pop最小元素栈
        时间复杂度：O(1)。对于这里的所有操作。
        空间复杂度：O(n)
*/
type MinStack struct {
    stack []int
    minStack []int
}

func Constructor() MinStack {
    return MinStack{
        stack: []int{},
        minStack: []int{math.MaxInt64},
    }
}

func (this *MinStack) Push(val int)  {
    this.stack = append(this.stack,val)
    minStackTop := this.minStack[len(this.minStack)-1]
    this.minStack = append(this.minStack,min(val,minStackTop))
}


func (this *MinStack) Pop()  {
    this.stack = this.stack[:len(this.stack)-1]
    this.minStack = this.minStack[:len(this.minStack)-1]
}


func (this *MinStack) Top() int {
    return this.stack[len(this.stack)-1]
}


func (this *MinStack) GetMin() int {
    return this.minStack[len(this.minStack)-1]
}

func min(x,y int) int{
    if x < y{
        return x
    }else{
        return y
    }
}


/**
 * Your MinStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(val);
 * obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.GetMin();
 */
```

4. 逆波兰表达式求值

```go
func evalRPN(tokens []string) int {
    /*思路：用栈来处理，遇到数字压栈，遇到运算符出栈两次进行运算，
    再压栈，最后返回栈中唯一的数字即可。需要注意的是遍历字符串数组时，
    将字符串转为int用func Atoi(s string) (int, error){}。其中返回
    两个参数，第一个是转化的数字，第二个是是否发生转化错误，若error为
    nil，则说明转化成功，若不为nil，则转化失败
    */
    stack1 := Constructor()
    for _,val := range tokens{
        number,err := strconv.Atoi(val)
        if err == nil{
            stack1.push(number)
        }else if val == "+"{
            temp1 := stack1.pop()
            temp2 := stack1.pop()
            temp3 := temp2 + temp1
            stack1.push(temp3)
        }else if val == "-"{
            temp1 := stack1.pop()
            temp2 := stack1.pop()
            temp3 := temp2 - temp1
            stack1.push(temp3)
        }else if val == "*"{
            temp1 := stack1.pop()
            temp2 := stack1.pop()
            temp3 := temp2 * temp1
            stack1.push(temp3)
        }else if val == "/"{
            temp1 := stack1.pop()
            temp2 := stack1.pop()
            temp3 := temp2 / temp1
            stack1.push(temp3)
        }
    }
    return stack1.pop()
}
type stack struct{
    data []int
}
func Constructor() stack{
    return stack{
        data: []int{},
    }
}
func (this *stack) pop() int{
    top := this.data[len(this.data)-1]
    this.data = this.data[:len(this.data)-1]
    return top
}
func (this *stack) push(val int){
    this.data = append(this.data,val)
}
```

5. 基本计算器

```go
func calculate(s string) int {
    /*思路：根据题意，只有加减运算，设置一个sign用于表示当前要计算数字是正是负，
    用栈来处理括号。当扫描到数字时，由于有多位数的可能，因此对后面的位进行遍历，
    遍历到不是数字时，将当前数字进行运算。若扫描到加减号，则改变sign，若扫描到'('，
    则将当前的计算结果和符号压栈，当扫描到')'时，将当前的res与以前的res进行计算。
    解释一下res的计算逻辑和压栈出栈的逻辑。如(4+5+2)，首先将0和1压栈，代表之前的
    结果是0，符号为1，然后将res置0，sign置1，计算内层4+5+2，最后弹出之前的符号和
    结果，进行运算。本来还可以先转后缀表达式再计算的，但是用range访问string再转化
    为byte太麻烦了，就不写了，感觉以后访问string还得是用下标i，用range反而难写，
    其次是还要注意byte转int时要用int(byte-'0')来转，不减'0'的话得到的数字是错的，
    这是因为ASCILL码中'0'的ASCILL码为48。
    时间复杂度：O(n)
    空间复杂度：O(n)
    */
    stack := []int{}
    sign := 1
    res := 0
    num := 0
    for i := 0;i < len(s);i++{
        if s[i] >= '0' && s[i] <= '9'{
            j := i
            num = 0
            for j < len(s) && s[j] >= '0' && s[j] <= '9'{
                num = num*10 + int(s[j]-'0')
                j++
            }
            res += sign*num
            i = j - 1
        }else if s[i] == '+'{
            sign = 1
        }else if s[i] == '-'{
            sign = -1
        }else if s[i] == '('{
            stack = append(stack,res)
            stack = append(stack,sign)
            res = 0
            sign = 1
        }else if s[i] == ')'{
            sign = stack[len(stack)-1]
            prevRes := stack[len(stack)-2]
            stack = stack[:len(stack)-2]
            res = prevRes + sign*res
        }
    }
    return res
}
```

### 链表

1. 环形链表

```go

```

2. 两数相加

```go

```

3. 合并两个有序链表

```go

```

4. 随机链表的复制

```go

```

5. 反转链表II

```go

```

6. K个一组反转链表

```go

```

7. 删除链表的倒数第N个结点

```go

```

8. 删除排序链表中的重复元素II

```go

```

9. 旋转链表

```go

```

10. 分隔链表

```go

```

11. LRU缓存

```go

```

### 二叉树

1. 二叉树的最大深度

```go

```

2. 相同的树

```go

```

3. 翻转二叉树

```go

```

4. 对称二叉树

```go

```

5. 从前序与中序遍历序列构造二叉树

```go

```

6. 从中序与后序遍历序列构造二叉树

```go

```

7. 填充每个节点的下一个右侧节点指针II

```go

```

8. 二叉树展开为链表

```go

```

9. 路径总和

```go

```

10. 求根节点到叶节点数字之和

```go

```

11. 二叉树中的最大路径和

```go

```

12. 二叉搜索树迭代器

```go

```

13. 完全二叉树的节点个数

```go

```

14. 二叉树的最近公共祖先

```go

```

### 二叉树层次遍历

1. 二叉树的右视图

```go

```

2. 二叉树的层平均值

```go

```

3. 二叉树的层序遍历

```go

```

4. 二叉树的锯齿形层序遍历

```go

```

### 二叉搜索树

1. 二叉搜索树的最小绝对差

```go

```

2. 二叉搜索树中第K小的元素

```go

```

3. 验证二叉搜索树

```go

```

