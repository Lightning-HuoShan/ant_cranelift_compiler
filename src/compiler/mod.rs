pub mod arc;
pub mod compile_state_impl;
pub mod compiler_impl;
pub mod function;
pub mod generic;
pub mod handler;
pub mod table;

mod constants;
mod convert_type;
mod imm;

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use ant_crate_def::Crate;
use ant_id::DefId;
use ant_ty::TyId;
use ant_typed_module::module::TypedModule;
use cranelift::prelude::Type;
use cranelift_codegen::{isa::TargetIsa, settings};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataId, FuncId};
use cranelift_object::ObjectModule;
use indexmap::IndexMap;

use crate::compiler::generic::{CompiledGenericInfo, GenericInfo};
use crate::compiler::table::SymbolTable;

use crate::args::read_arg;
use crate::link_utils::{TargetTriple, build_static_library, get_compiler_dir, get_linker_config, get_target_triple_or_default, link_executable};

pub type CompileResult<T> = Result<T, String>;

// 编译器结构体
pub struct Compiler<'a> {
    module: ObjectModule,

    builder_ctx: FunctionBuilderContext,
    context: cranelift_codegen::Context,

    function_map: HashMap<String, cranelift_module::FuncId>,
    data_map: HashMap<String, cranelift_module::DataId>,
    generic_map: HashMap<String, GenericInfo>,
    compiled_generic_map: IndexMap<String, CompiledGenericInfo>,
    def_functions: HashMap<DefId, cranelift_module::FuncId>,
    def_datas: HashMap<DefId, cranelift_module::DataId>,

    target_isa: Arc<dyn TargetIsa>,

    table: Rc<RefCell<SymbolTable>>,
    typed_module: TypedModule<'a>,
    krate: Crate<'a>,

    arc_alloc: FuncId,
    arc_retain: FuncId,
    arc_release: FuncId,

    ptr_type: Type,
}

pub struct GlobalState<'a, 'b> {
    pub target_isa: Arc<dyn TargetIsa>,
    pub module: &'a mut ObjectModule,

    pub function_map: &'a mut HashMap<String, cranelift_module::FuncId>,
    pub data_map: &'a mut HashMap<String, cranelift_module::DataId>,
    pub generic_map: &'a mut HashMap<String, GenericInfo>,
    pub compiled_generic_map: &'a mut IndexMap<String, CompiledGenericInfo>,
    pub def_functions: &'a mut HashMap<DefId, cranelift_module::FuncId>,
    pub def_datas: &'a mut HashMap<DefId, cranelift_module::DataId>,

    pub table: Rc<RefCell<SymbolTable>>,

    pub typed_module: &'a mut TypedModule<'b>,

    pub krate: &'a mut Crate<'b>,

    pub arc_alloc: FuncId,
    pub arc_retain: FuncId,
    pub arc_release: FuncId,

    pub ptr_type: Type,
}

pub struct FunctionState<'a, 'b> {
    pub builder: FunctionBuilder<'a>,
    pub target_isa: Arc<dyn TargetIsa>,
    pub module: &'a mut ObjectModule,

    pub function_map: &'a mut HashMap<String, cranelift_module::FuncId>,
    pub data_map: &'a mut HashMap<String, cranelift_module::DataId>,
    pub generic_map: &'a mut HashMap<String, GenericInfo>,
    pub compiled_generic_map: &'a mut IndexMap<String, CompiledGenericInfo>,
    pub subst: &'a IndexMap<Arc<str>, TyId>,
    pub def_functions: &'a mut HashMap<DefId, cranelift_module::FuncId>,
    pub def_datas: &'a mut HashMap<DefId, cranelift_module::DataId>,

    pub krate: &'a mut Crate<'b>,

    pub typed_module: &'a mut TypedModule<'b>,

    pub table: Rc<RefCell<SymbolTable>>,

    pub arc_alloc: FuncId,
    pub arc_retain: FuncId,
    pub arc_release: FuncId,

    pub terminated: bool,
    pub ptr_type: Type,
}

#[allow(unused)]
pub trait CompileState<'a, 'b> {
    fn get_target_isa(&self) -> Arc<dyn TargetIsa>;
    fn get_module(&mut self) -> &mut ObjectModule;
    fn get_function_map(&mut self) -> &mut HashMap<String, cranelift_module::FuncId>;
    fn get_data_map(&mut self) -> &mut HashMap<String, cranelift_module::DataId>;
    fn get_generic_map(&mut self) -> &mut HashMap<String, GenericInfo>;
    fn get_compiled_generic_map(&mut self) -> &mut IndexMap<String, CompiledGenericInfo>;
    fn get_typed_module(&'b mut self) -> &'a mut TypedModule<'b>;
    fn get_typed_module_ref(&self) -> &TypedModule<'_>;
    fn get_def_functions(&mut self) -> &mut HashMap<DefId, FuncId>;
    fn get_def_datas(&mut self) -> &mut HashMap<DefId, DataId>;
    fn get_krate(&'b mut self) -> &'a mut Crate<'b>;
    fn get_krate_ref(&'_ self) -> &'_ Crate<'_>;

    fn get_table(&self) -> Rc<RefCell<SymbolTable>>;

    fn get_arc_alloc(&self) -> FuncId;
    fn get_arc_retain(&self) -> FuncId;
    fn get_arc_release(&self) -> FuncId;
}

// 创建目标 ISA 的辅助函数
pub fn create_target_isa() -> Arc<dyn TargetIsa> {
    let flag_builder = settings::builder();
    // flag_builder.set("opt_level", "speed").unwrap();

    let isa_builder = cranelift_native::builder().unwrap();
    isa_builder
        .finish(settings::Flags::new(flag_builder))
        .unwrap()
}

/// 将对象代码编译为可执行文件
///
/// - `object_code`: 已编译的原始对象代码（例如来自 LLVM 的 MC 层输出）
/// - `output_path`: 最终可执行文件的完整路径（包含文件名和后缀）
///
/// 如果 `--compile-only` 被指定，则只生成 `.o` 目标文件并停止。
/// 如果 `--keep-cache` 被指定，则 `.o` 文件会保留在输出文件所在目录中。
/// 否则，`.o` 文件将使用临时文件，链接完成后会自动删除。
pub fn compile_to_executable(
    object_code: &[u8],
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    use std::path::Path;

    let args = read_arg();

    // 确定目标文件（.o）的存放位置
    let keep_cache = args.as_ref().map(|a| a.keep_cache).unwrap_or(false);
    let compile_only = args.as_ref().map(|a| a.compile_only).unwrap_or(false);

    let (object_file_path, _temp_dir_guard) = if keep_cache || compile_only {
        // 保留 .o 文件：放在与最终输出相同的目录中
        let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
        let stem = output_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let obj_path = parent.join(format!("{stem}.o"));
        (obj_path, None)
    } else {
        // 使用临时目录，函数结束时自动删除
        let temp_dir = tempfile::tempdir()?;
        let obj_path = temp_dir.path().join("output.o");
        (obj_path, Some(temp_dir))
    };

    // 确保输出目录存在（包括 .o 文件的父目录和最终可执行文件的父目录）
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = object_file_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // 写入目标文件
    fs::write(&object_file_path, object_code)?;

    // 如果仅需编译，到此为止
    if compile_only {
        return Ok(());
    }

    // 获取目标三元组
    let target = get_target_triple_or_default(&args);
    let target_triple = TargetTriple::new(&target);

    // 将 .o 打包为静态库（这一步骤对不同平台会生成 .a 或 .lib）
    let lib_path = build_static_library(&object_file_path, output_path, &target, &args)?;

    // 准备链接器配置
    let linker_config = get_linker_config(&target_triple)?;
    let compiler_dir = get_compiler_dir();

    // 执行最终链接
    link_executable(
        &target_triple,
        &lib_path,
        output_path,
        &compiler_dir,
        &linker_config,
        &args,
    )?;

    // 清理临时静态库（它位于输出目录，命名为 lib<stem>.<a/lib>）
    let _ = fs::remove_file(&lib_path);

    Ok(())
}

pub fn get_platform_width() -> usize {
    #[cfg(target_pointer_width = "64")]
    return 64;

    #[cfg(target_pointer_width = "32")]
    return 32;

    #[cfg(target_pointer_width = "16")]
    return 16;
}

impl<'a, 'b> CompileState<'a, 'b> for GlobalState<'a, 'b> {
    fn get_target_isa(&self) -> Arc<dyn TargetIsa> {
        self.target_isa.clone()
    }

    fn get_module(&mut self) -> &mut ObjectModule {
        self.module
    }

    fn get_function_map(&mut self) -> &mut HashMap<String, cranelift_module::FuncId> {
        self.function_map
    }

    fn get_data_map(&mut self) -> &mut HashMap<String, cranelift_module::DataId> {
        self.data_map
    }

    fn get_generic_map(&mut self) -> &mut HashMap<String, GenericInfo> {
        self.generic_map
    }

    fn get_compiled_generic_map(&mut self) -> &mut IndexMap<String, CompiledGenericInfo> {
        self.compiled_generic_map
    }

    fn get_table(&self) -> Rc<RefCell<SymbolTable>> {
        self.table.clone()
    }

    fn get_typed_module(&'b mut self) -> &'a mut TypedModule<'b> {
        self.typed_module
    }

    fn get_typed_module_ref(&self) -> &TypedModule<'_> {
        self.typed_module
    }

    fn get_arc_alloc(&self) -> FuncId {
        self.arc_alloc
    }

    fn get_arc_retain(&self) -> FuncId {
        self.arc_retain
    }

    fn get_arc_release(&self) -> FuncId {
        self.arc_release
    }

    fn get_def_functions(&mut self) -> &mut HashMap<DefId, FuncId> {
        self.def_functions
    }

    fn get_def_datas(&mut self) -> &mut HashMap<DefId, DataId> {
        self.def_datas
    }

    fn get_krate(&'b mut self) -> &'a mut Crate<'b> {
        self.krate
    }

    fn get_krate_ref(&'_ self) -> &'_ Crate<'_> {
        self.krate
    }
}

impl<'a, 'b> CompileState<'a, 'b> for FunctionState<'a, 'b> {
    fn get_target_isa(&self) -> Arc<dyn TargetIsa> {
        self.target_isa.clone()
    }

    fn get_module(&mut self) -> &mut ObjectModule {
        self.module
    }

    fn get_function_map(&mut self) -> &mut HashMap<String, cranelift_module::FuncId> {
        self.function_map
    }

    fn get_data_map(&mut self) -> &mut HashMap<String, cranelift_module::DataId> {
        self.data_map
    }

    fn get_generic_map(&mut self) -> &mut HashMap<String, GenericInfo> {
        self.generic_map
    }

    fn get_compiled_generic_map(&mut self) -> &mut IndexMap<String, CompiledGenericInfo> {
        self.compiled_generic_map
    }

    fn get_table(&self) -> Rc<RefCell<SymbolTable>> {
        self.table.clone()
    }

    fn get_typed_module(&'b mut self) -> &'a mut TypedModule<'b> {
        self.typed_module
    }

    fn get_typed_module_ref(&'_ self) -> &'_ TypedModule<'_> {
        self.typed_module
    }

    fn get_arc_alloc(&self) -> FuncId {
        self.arc_alloc
    }

    fn get_arc_retain(&self) -> FuncId {
        self.arc_retain
    }

    fn get_arc_release(&self) -> FuncId {
        self.arc_release
    }

    fn get_def_functions(&mut self) -> &mut HashMap<DefId, FuncId> {
        self.def_functions
    }

    fn get_def_datas(&mut self) -> &mut HashMap<DefId, DataId> {
        self.def_datas
    }

    fn get_krate(&'b mut self) -> &'a mut Crate<'b> {
        self.krate
    }

    fn get_krate_ref(&self) -> &Crate<'_> {
        self.krate
    }
}
